"""
DIRA — Dataset Builder
======================
Iterates a candidate VCF (from bcftools / FreeBayes / GATK HC),
cross-references a GIAB truth VCF to assign labels,
extracts features for every indel, and writes a labelled
Parquet/CSV dataset ready for model training.

Usage:
    python build_dataset.py \
        --candidate  chr20_indels.vcf.gz \
        --truth      HG002_chr20_indels_truth.vcf.gz \
        --bam        HG002_chr20.bam \
        --ref        GRCh38.fa \
        --out        chr20_features.parquet \
        [--region    chr20]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cyvcf2

from features import DIRAFeatureExtractor, FEATURE_NAMES, NEARBY_VAR_WINDOW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dira.build")


# ── Truth-set index ───────────────────────────────────────────────────────────

def build_truth_index(truth_vcf_path: str) -> set:
    """
    Return a set of (chrom, pos, ref, alt) tuples from the truth VCF.
    Only PASS / no-filter records are included.
    Positions are 1-based to match VCF convention.
    """
    log.info(f"Indexing truth VCF: {truth_vcf_path}")
    truth = set()
    vcf = cyvcf2.VCF(truth_vcf_path)
    for v in vcf:
        if v.FILTER and v.FILTER != "PASS":
            continue
        for alt in v.ALT:
            # Only indels
            if len(v.REF) != len(alt):
                truth.add((v.CHROM, v.POS, v.REF, alt))
    vcf.close()
    log.info(f"Truth index: {len(truth):,} indel records")
    return truth


# ── Nearby variant density ────────────────────────────────────────────────────

def build_variant_density(candidate_vcf_path: str) -> dict:
    """
    For each candidate indel, count how many other candidates fall
    within NEARBY_VAR_WINDOW bp. Returns dict keyed by (chrom, pos).
    O(N log N) via sorted position list per chromosome.
    """
    from collections import defaultdict
    import bisect

    log.info("Computing nearby variant density ...")
    positions = defaultdict(list)
    vcf = cyvcf2.VCF(candidate_vcf_path)
    for v in vcf:
        for alt in v.ALT:
            if len(v.REF) != len(alt):
                positions[v.CHROM].append(v.POS)
    vcf.close()

    density = {}
    for chrom, pos_list in positions.items():
        pos_list.sort()
        arr = np.array(pos_list)
        for p in pos_list:
            lo = bisect.bisect_left(arr, p - NEARBY_VAR_WINDOW)
            hi = bisect.bisect_right(arr, p + NEARBY_VAR_WINDOW)
            # subtract 1 to exclude self
            density[(chrom, p)] = (hi - lo) - 1
    return density


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_dataset(
    candidate_vcf: str,
    truth_vcf: str,
    bam_path: str,
    ref_path: str,
    out_path: str,
    region: str = None,
):
    truth_index   = build_truth_index(truth_vcf)
    var_density   = build_variant_density(candidate_vcf)

    records = []
    skipped = 0
    n_pos   = 0
    n_neg   = 0

    log.info(f"Extracting features from {candidate_vcf}")

    with DIRAFeatureExtractor(bam_path, ref_path) as extractor:
        vcf = cyvcf2.VCF(candidate_vcf)

        if region:
            vcf = vcf(region)

        for v in vcf:
            # Only process indels
            for alt in v.ALT:
                if len(v.REF) == len(alt):
                    continue  # skip SNPs

                try:
                    nearby = var_density.get((v.CHROM, v.POS), 0)
                    feat   = extractor.extract(
                        chrom=v.CHROM,
                        pos=v.POS,
                        ref_allele=v.REF,
                        alt_allele=alt,
                        nearby_variant_count=nearby,
                    )
                except Exception as e:
                    log.warning(f"Skipped {v.CHROM}:{v.POS} {v.REF}>{alt} — {e}")
                    skipped += 1
                    continue

                label = int((v.CHROM, v.POS, v.REF, alt) in truth_index)

                row = {
                    "chrom":    v.CHROM,
                    "pos":      v.POS,
                    "ref":      v.REF,
                    "alt":      alt,
                    "label":    label,
                    "qual":     float(v.QUAL) if v.QUAL is not None else 0.0,
                }
                for fname, fval in zip(FEATURE_NAMES, feat):
                    row[fname] = fval

                records.append(row)
                n_pos += label
                n_neg += (1 - label)

        vcf.close()

    log.info(
        f"Extracted {len(records):,} indels  "
        f"(TP={n_pos:,}, FP={n_neg:,}, skipped={skipped:,})"
    )
    log.info(f"Class ratio (neg/pos) = {n_neg / max(n_pos, 1):.3f}  "
             f"→ use scale_pos_weight = {n_neg / max(n_pos, 1):.3f}")

    df = pd.DataFrame(records)

    out = Path(out_path)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

    log.info(f"Saved → {out_path}  ({out.stat().st_size / 1024:.1f} KB)")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="DIRA dataset builder")
    p.add_argument("--candidate", required=True, help="Candidate indel VCF (gz+tbi)")
    p.add_argument("--truth",     required=True, help="GIAB truth VCF (gz+tbi)")
    p.add_argument("--bam",       required=True, help="Indexed BAM file")
    p.add_argument("--ref",       required=True, help="Reference FASTA (indexed)")
    p.add_argument("--out",       required=True, help="Output path (.parquet or .csv)")
    p.add_argument("--region",    default=None,  help="Optional region, e.g. chr20")
    args = p.parse_args()

    build_dataset(
        candidate_vcf=args.candidate,
        truth_vcf=args.truth,
        bam_path=args.bam,
        ref_path=args.ref,
        out_path=args.out,
        region=args.region,
    )


if __name__ == "__main__":
    main()
