#!/usr/bin/env python3
"""
Feature Extractor v3 — Parallelized & Optimized
=================================================
Parallelizes by chromosome using multiprocessing.
Optimized hot paths: set-based read lookup, pure-python stats,
early-exit NBQ, and batched CSV writes.

Target: GCP L2 (n2-standard-8), chr1–chr18.
"""

import pysam
import math
import os
import sys
import csv
import argparse
import tempfile
import shutil
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial


# ---------------------------------------------------------------------------
# Pure-python stats (avoids numpy overhead on small arrays)
# ---------------------------------------------------------------------------

def _mean(arr):
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def _var(arr):
    if not arr:
        return 0.0
    n = len(arr)
    if n == 1:
        return 0.0
    m = sum(arr) / n
    return sum((x - m) ** 2 for x in arr) / n


def shannon_entropy(arr):
    n = len(arr)
    if n == 0:
        return 0.0
    counts = Counter(arr)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


# ---------------------------------------------------------------------------
# NBQ with early exit past the window
# ---------------------------------------------------------------------------

def compute_nbq(read, var_pos, window=5):
    """
    Neighbourhood Base Quality — average BQ within ±window of var_pos
    in reference coordinates.  Early-exits once past the window.
    """
    quals = read.query_qualities
    if quals is None:
        return 0.0

    total = 0
    count = 0
    lo = var_pos - window
    hi = var_pos + window

    for qpos, rpos in read.get_aligned_pairs(matches_only=True):
        if rpos is None:
            continue
        if rpos < lo:
            continue
        if rpos > hi:
            break                       # aligned pairs are sorted — safe to stop
        total += quals[qpos]
        count += 1

    return (total / count) if count else 0.0


# ---------------------------------------------------------------------------
# CSV header (shared across workers)
# ---------------------------------------------------------------------------

HEADER = [
    "chrom", "pos", "ref", "alt",
    "allele_balance", "total_depth", "alt_depth",
    "mq_mean_alt", "mq_var_alt", "mq_mean_ref",
    "strand_ratio_alt", "strand_ratio_ref",
    "read_end_dist_mean", "read_end_dist_var",
    "softclip_rate_alt", "softclip_mean_alt",
    "bq_mean_alt", "bq_var_alt", "bq_mean_ref",
    "nbq_alt", "nbq_ref",
    "bq_drop_alt",
    "indel_length", "is_insertion",
    "insert_entropy",
    "homopolymer_length", "gc_content_50bp",
    "udp_dp_ratio", "pileup_entropy",
]


# ---------------------------------------------------------------------------
# Worker: process all indels on one chromosome
# ---------------------------------------------------------------------------

def process_chromosome(chrom, vcf_path, bam_path, ref_path, tmp_dir):
    """
    Extract features for every indel on *chrom*.
    Writes to a temp CSV; returns the path.
    """
    from cyvcf2 import VCF          # import per-worker (cyvcf2 is not fork-safe)

    vcf = VCF(vcf_path)
    bam = pysam.AlignmentFile(bam_path, "rb")
    ref = pysam.FastaFile(ref_path)

    tmp_path = os.path.join(tmp_dir, f"{chrom}.csv")
    rows_buf = []
    FLUSH_EVERY = 500                # batch writes

    def flush():
        nonlocal rows_buf
        if not rows_buf:
            return
        with open(tmp_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(rows_buf)
        rows_buf = []

    # Restrict iteration to this chromosome
    try:
        var_iter = vcf(chrom)
    except Exception:
        # If the VCF has no records for this chrom, just return empty
        bam.close()
        ref.close()
        return tmp_path

    for var in var_iter:

        # Only indels
        if var.ALT is None or len(var.ALT) == 0:
            continue
        if len(var.REF) == len(var.ALT[0]):
            continue

        pos = var.POS          # 1-based
        alt = var.ALT[0]
        ref_allele = var.REF
        pos0 = pos - 1        # 0-based for pysam

        # ------------------------------------------------------------------
        # Collect reads at the variant site
        # ------------------------------------------------------------------
        reads = []
        bases = []
        alt_reads_set = set()   # store indices for O(1) lookup
        alt_reads = []
        ref_reads = []

        for col in bam.pileup(chrom, pos0, pos0 + 1,
                               truncate=True,
                               min_base_quality=0,
                               min_mapping_quality=0):
            if col.pos != pos0:
                continue

            for pr in col.pileups:
                if pr.is_del or pr.is_refskip:
                    continue

                r = pr.alignment
                qpos = pr.query_position
                if qpos is None:
                    continue
                base = r.query_sequence[qpos]

                idx = len(reads)
                reads.append((r, qpos))
                bases.append(base)

                if base == alt[0]:
                    alt_reads.append((r, qpos))
                    alt_reads_set.add(idx)
                else:
                    ref_reads.append((r, qpos))

        depth = len(reads)
        alt_depth = len(alt_reads)
        if depth == 0:
            continue

        allele_balance = alt_depth / depth

        # ------------------------------------------------------------------
        # Mapping quality
        # ------------------------------------------------------------------
        mq_alt = [r.mapping_quality for r, _ in alt_reads]
        mq_ref = [r.mapping_quality for r, _ in ref_reads]

        mq_mean_alt = _mean(mq_alt)
        mq_var_alt  = _var(mq_alt)
        mq_mean_ref = _mean(mq_ref)

        # ------------------------------------------------------------------
        # Strand ratios
        # ------------------------------------------------------------------
        strand_alt = sum(1 for r, _ in alt_reads if not r.is_reverse)
        strand_ratio_alt = strand_alt / alt_depth if alt_depth else 0.0

        strand_ref = sum(1 for r, _ in ref_reads if not r.is_reverse)
        strand_ratio_ref = strand_ref / len(ref_reads) if ref_reads else 0.0

        # ------------------------------------------------------------------
        # Read-end distance & NBQ (single pass over all reads)
        # ------------------------------------------------------------------
        read_end_dist = []
        nbq_alt_vals = []
        nbq_ref_vals = []

        for i, (r, qpos) in enumerate(reads):
            rlen = r.query_length or r.infer_read_length() or 150
            read_end_dist.append(min(qpos, rlen - qpos))

            nbq = compute_nbq(r, pos)

            if i in alt_reads_set:
                nbq_alt_vals.append(nbq)
            else:
                nbq_ref_vals.append(nbq)

        read_end_dist_mean = _mean(read_end_dist)
        read_end_dist_var  = _var(read_end_dist)

        nbq_alt_mean = _mean(nbq_alt_vals)
        nbq_ref_mean = _mean(nbq_ref_vals)

        # ------------------------------------------------------------------
        # Base quality (whole-read average per read)
        # ------------------------------------------------------------------
        def _read_mean_bq(r):
            q = r.query_qualities
            if q is None or len(q) == 0:
                return 0.0
            return sum(q) / len(q)

        bq_alt = [_read_mean_bq(r) for r, _ in alt_reads]
        bq_ref = [_read_mean_bq(r) for r, _ in ref_reads]

        bq_mean_alt = _mean(bq_alt)
        bq_var_alt  = _var(bq_alt)
        bq_mean_ref = _mean(bq_ref)

        bq_drop_alt = bq_mean_ref - bq_mean_alt

        # ------------------------------------------------------------------
        # Soft-clip stats (alt reads only)
        # ------------------------------------------------------------------
        softclips = []
        for r, _ in alt_reads:
            ct = r.cigartuples
            if ct:
                sc = sum(length for op, length in ct if op == 4)
            else:
                sc = 0
            softclips.append(sc)

        softclip_rate_alt = _mean(softclips)
        softclip_mean_alt = softclip_rate_alt

        # ------------------------------------------------------------------
        # Indel properties
        # ------------------------------------------------------------------
        indel_length = len(alt) - len(ref_allele)
        is_insertion = 1 if indel_length > 0 else 0
        insert_entropy = shannon_entropy(alt) if is_insertion else 0.0

        # ------------------------------------------------------------------
        # Reference context (homopolymer + GC)
        # ------------------------------------------------------------------
        ctx_start = max(0, pos0 - 50)
        ctx_end   = pos0 + 50
        seq = ref.fetch(chrom, ctx_start, ctx_end)

        if len(seq) == 0:
            gc = 0.0
            homopoly = 1
        else:
            gc = (seq.count("G") + seq.count("C")) / len(seq)

            center_idx = pos0 - ctx_start
            if center_idx < 0 or center_idx >= len(seq):
                homopoly = 1
            else:
                center = seq[center_idx]
                homopoly = 1
                j = center_idx - 1
                while j >= 0 and seq[j] == center:
                    homopoly += 1
                    j -= 1
                j = center_idx + 1
                while j < len(seq) and seq[j] == center:
                    homopoly += 1
                    j += 1

        # ------------------------------------------------------------------
        # UDP / DP ratio
        # ------------------------------------------------------------------
        udp = depth
        dp = sum(1 for r, _ in reads if r.mapping_quality >= 20)
        udp_dp = udp / dp if dp > 0 else 0.0

        # ------------------------------------------------------------------
        # Pileup entropy
        # ------------------------------------------------------------------
        pile_entropy = shannon_entropy(bases)

        # ------------------------------------------------------------------
        # Append row
        # ------------------------------------------------------------------
        rows_buf.append([
            chrom, pos, ref_allele, alt,
            allele_balance, depth, alt_depth,
            mq_mean_alt, mq_var_alt, mq_mean_ref,
            strand_ratio_alt, strand_ratio_ref,
            read_end_dist_mean, read_end_dist_var,
            softclip_rate_alt, softclip_mean_alt,
            bq_mean_alt, bq_var_alt, bq_mean_ref,
            nbq_alt_mean, nbq_ref_mean,
            bq_drop_alt,
            indel_length, is_insertion,
            insert_entropy,
            homopoly, gc,
            udp_dp, pile_entropy,
        ])

        if len(rows_buf) >= FLUSH_EVERY:
            flush()

    flush()
    bam.close()
    ref.close()

    return tmp_path


# ---------------------------------------------------------------------------
# Merge per-chromosome CSVs into the final output
# ---------------------------------------------------------------------------

def merge_csvs(tmp_dir, chroms, out_csv):
    with open(out_csv, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(HEADER)

        for chrom in chroms:
            tmp_path = os.path.join(tmp_dir, f"{chrom}.csv")
            if not os.path.exists(tmp_path):
                continue
            with open(tmp_path, "r") as fin:
                for line in fin:
                    fout.write(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallelized indel feature extractor (chr1-chr18)")
    parser.add_argument("--vcf", required=True, help="Input VCF (bgzipped + tabix-indexed)")
    parser.add_argument("--bam", required=True, help="Input BAM (indexed)")
    parser.add_argument("--ref", required=True, help="Reference FASTA (indexed)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--chroms", nargs="+",
                        default=[f"chr{i}" for i in range(1, 19)],
                        help="Chromosomes to process (default: chr1-chr18)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: auto = cpu_count)")
    args = parser.parse_args()

    chroms = args.chroms
    n_workers = args.workers if args.workers > 0 else min(len(chroms), cpu_count())

    print(f"[feature_extractor_v3] Processing {len(chroms)} chromosomes "
          f"with {n_workers} workers", file=sys.stderr)

    # Temp directory for per-chrom CSVs
    tmp_dir = tempfile.mkdtemp(prefix="feat_extract_")

    try:
        # Build argument tuples — one per chromosome
        worker_fn = partial(
            _worker_wrapper,
            vcf_path=args.vcf,
            bam_path=args.bam,
            ref_path=args.ref,
            tmp_dir=tmp_dir,
        )

        with Pool(processes=n_workers) as pool:
            pool.map(worker_fn, chroms)

        merge_csvs(tmp_dir, chroms, args.out)
        print(f"[feature_extractor_v3] Done → {args.out}", file=sys.stderr)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _worker_wrapper(chrom, vcf_path, bam_path, ref_path, tmp_dir):
    """Top-level picklable wrapper for Pool.map."""
    print(f"  [{chrom}] started", file=sys.stderr)
    result = process_chromosome(chrom, vcf_path, bam_path, ref_path, tmp_dir)
    print(f"  [{chrom}] done", file=sys.stderr)
    return result


if __name__ == "__main__":
    main()
