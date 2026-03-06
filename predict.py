#!/usr/bin/env python3
"""
DIRA — Inference v3
====================
Parallelized prediction pipeline aligned with feature_extractor_v3 and
train_dira_model_v3. Uses the EXACT same feature extraction code path
as training to guarantee feature parity.

Key changes from v2:
  1. Feature extraction uses the same pileup logic as feature_extractor_v3
     — no separate features.py dependency.
  2. Parallelized by chromosome (same multiprocessing pattern as extractor).
  3. Feature names/order validated against features.json saved during training.
  4. Platt scaler always loaded (training v3 always saves it).
  5. Per-variant VCF annotation with DIRA_PROB, recalibrated QUAL, FILTER.

Usage:
    python predict_dira_v3.py \\
        --candidate  candidates.vcf.gz \\
        --bam        sample.bam \\
        --ref        GRCh38.fa \\
        --model      model_v3/ \\
        --out        filtered.vcf \\
        [--chroms chr1 chr2 ...] \\
        [--workers 8]
"""

import argparse
import csv
import json
import logging
import math
import os
import pickle
import sys
import tempfile
import shutil
from collections import Counter
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pysam
import lightgbm as lgb
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dira.predict")


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction — IDENTICAL to feature_extractor_v3
# This is duplicated intentionally so predict has zero external dependencies
# beyond standard libs + pysam + lightgbm + cyvcf2.
# If you change feature_extractor_v3.py, you MUST update this section too.
# ══════════════════════════════════════════════════════════════════════════════

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


def compute_nbq(read, var_pos, window=5):
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
            break
        total += quals[qpos]
        count += 1
    return (total / count) if count else 0.0


def extract_features_at_site(bam, ref, chrom, pos, ref_allele, alt_allele):
    """
    Extract the feature vector for a single indel site.
    Returns a dict of feature_name -> value, or None if depth == 0.

    Logic is identical to feature_extractor_v3.process_chromosome.
    """
    pos0 = pos - 1

    reads = []
    bases = []
    alt_reads_set = set()
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

            if base == alt_allele[0]:
                alt_reads.append((r, qpos))
                alt_reads_set.add(idx)
            else:
                ref_reads.append((r, qpos))

    depth = len(reads)
    alt_depth = len(alt_reads)
    if depth == 0:
        return None

    allele_balance = alt_depth / depth

    # Mapping quality
    mq_alt = [r.mapping_quality for r, _ in alt_reads]
    mq_ref = [r.mapping_quality for r, _ in ref_reads]
    mq_mean_alt = _mean(mq_alt)
    mq_var_alt = _var(mq_alt)
    mq_mean_ref = _mean(mq_ref)

    # Strand ratios
    strand_alt = sum(1 for r, _ in alt_reads if not r.is_reverse)
    strand_ratio_alt = strand_alt / alt_depth if alt_depth else 0.0
    strand_ref = sum(1 for r, _ in ref_reads if not r.is_reverse)
    strand_ratio_ref = strand_ref / len(ref_reads) if ref_reads else 0.0

    # Read-end distance & NBQ
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
    read_end_dist_var = _var(read_end_dist)
    nbq_alt_mean = _mean(nbq_alt_vals)
    nbq_ref_mean = _mean(nbq_ref_vals)

    # Base quality
    def _read_mean_bq(r):
        q = r.query_qualities
        if q is None or len(q) == 0:
            return 0.0
        return sum(q) / len(q)

    bq_alt = [_read_mean_bq(r) for r, _ in alt_reads]
    bq_ref = [_read_mean_bq(r) for r, _ in ref_reads]
    bq_mean_alt = _mean(bq_alt)
    bq_var_alt = _var(bq_alt)
    bq_mean_ref = _mean(bq_ref)
    bq_drop_alt = bq_mean_ref - bq_mean_alt

    # Soft-clips
    softclips = []
    for r, _ in alt_reads:
        ct = r.cigartuples
        sc = sum(length for op, length in ct if op == 4) if ct else 0
        softclips.append(sc)
    softclip_rate_alt = _mean(softclips)

    # Indel properties
    indel_length = len(alt_allele) - len(ref_allele)
    is_insertion = 1 if indel_length > 0 else 0
    insert_entropy = shannon_entropy(alt_allele) if is_insertion else 0.0

    # Reference context
    ctx_start = max(0, pos0 - 50)
    ctx_end = pos0 + 50
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

    # UDP / DP ratio
    udp = depth
    dp = sum(1 for r, _ in reads if r.mapping_quality >= 20)
    udp_dp = udp / dp if dp > 0 else 0.0

    # Pileup entropy
    pile_entropy = shannon_entropy(bases)

    # Return as dict — order will be enforced by the feature list from training
    return {
        "allele_balance": allele_balance,
        "total_depth": depth,
        "alt_depth": alt_depth,
        "mq_mean_alt": mq_mean_alt,
        "mq_var_alt": mq_var_alt,
        "mq_mean_ref": mq_mean_ref,
        "strand_ratio_alt": strand_ratio_alt,
        "strand_ratio_ref": strand_ratio_ref,
        "read_end_dist_mean": read_end_dist_mean,
        "read_end_dist_var": read_end_dist_var,
        "softclip_rate_alt": softclip_rate_alt,
        "softclip_mean_alt": softclip_rate_alt,    # identical, kept for compat
        "bq_mean_alt": bq_mean_alt,
        "bq_var_alt": bq_var_alt,
        "bq_mean_ref": bq_mean_ref,
        "nbq_alt": nbq_alt_mean,
        "nbq_ref": nbq_ref_mean,
        "bq_drop_alt": bq_drop_alt,
        "indel_length": indel_length,
        "is_insertion": is_insertion,
        "insert_entropy": insert_entropy,
        "homopolymer_length": homopoly,
        "gc_content_50bp": gc,
        "udp_dp_ratio": udp_dp,
        "pileup_entropy": pile_entropy,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_dir):
    """
    Load LightGBM model, Platt scaler, threshold τ, and feature list.
    Validates that the feature list from training is present.
    """
    d = Path(model_dir)

    # ── LightGBM model ────────────────────────────────────────────────────────
    model_path = d / "lgbm_model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    log.info(f"Model loaded: {model.num_feature()} features, "
             f"best_iteration={model.best_iteration}")

    # ── Feature list (mandatory) ──────────────────────────────────────────────
    features_path = d / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(
            f"features.json not found in {model_dir}. "
            "Re-run training with train_dira_model_v3.py to generate it."
        )
    with open(features_path) as f:
        feature_names = json.load(f)

    if len(feature_names) != model.num_feature():
        raise ValueError(
            f"features.json has {len(feature_names)} features but model "
            f"expects {model.num_feature()}"
        )
    log.info(f"Feature list validated: {len(feature_names)} features")

    # ── Platt scaler ──────────────────────────────────────────────────────────
    platt = None
    platt_path = d / "platt_scaler.pkl"
    if platt_path.exists():
        with open(platt_path, "rb") as f:
            platt = pickle.load(f)
        log.info("Platt scaler loaded — using calibrated probabilities")
    else:
        log.warning("platt_scaler.pkl not found — falling back to raw sigmoid")

    # ── Threshold ─────────────────────────────────────────────────────────────
    tau = None
    for fname in ("results.json", "threshold.json"):
        p = d / fname
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            tau = data.get("threshold") or data.get("tau")
            if tau is not None:
                log.info(f"Threshold τ={tau:.4f} from {fname}")
                break

    if tau is None:
        tau = 0.5
        log.warning("No threshold found — defaulting to τ=0.5")

    return model, platt, float(tau), feature_names


# ══════════════════════════════════════════════════════════════════════════════
# Probability helpers
# ══════════════════════════════════════════════════════════════════════════════

def raw_to_prob(platt, raw_score):
    if platt is not None:
        return float(platt.predict_proba([[raw_score]])[0][1])
    return float(1.0 / (1.0 + math.exp(-raw_score)))


def prob_to_qual(prob):
    prob = min(prob, 1.0 - 1e-9)
    return -10.0 * math.log10(1.0 - prob)


# ══════════════════════════════════════════════════════════════════════════════
# Per-chromosome worker
# ══════════════════════════════════════════════════════════════════════════════

def predict_chromosome(chrom, vcf_path, bam_path, ref_path,
                       model_path, platt_path, feature_names,
                       tau, tmp_dir):
    """
    Run inference for all indels on one chromosome.
    Writes results to a temp TSV: chrom, pos, ref, alt, prob, qual, pass/filter.
    SNPs are written with prob=. (pass-through).
    """
    from cyvcf2 import VCF

    vcf = VCF(vcf_path)
    bam = pysam.AlignmentFile(bam_path, "rb")
    ref = pysam.FastaFile(ref_path)

    # Each worker loads its own copy of the model (not fork-safe)
    model = lgb.Booster(model_file=model_path)

    platt = None
    if platt_path and os.path.exists(platt_path):
        with open(platt_path, "rb") as f:
            platt = pickle.load(f)

    tmp_path = os.path.join(tmp_dir, f"{chrom}.tsv")
    buf = []
    FLUSH = 500

    n_pass = n_filter = n_skip = 0

    def flush():
        nonlocal buf
        if not buf:
            return
        with open(tmp_path, "a", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerows(buf)
        buf = []

    try:
        var_iter = vcf(chrom)
    except Exception:
        bam.close()
        ref.close()
        return tmp_path, 0, 0, 0

    for v in var_iter:
        for alt in v.ALT:
            is_indel = len(v.REF) != len(alt)

            if not is_indel:
                # SNP pass-through — mark with "." so merger knows
                buf.append([v.CHROM, v.POS, v.REF, alt, ".", ".", "SNP"])
                continue

            try:
                feat_dict = extract_features_at_site(
                    bam, ref, v.CHROM, v.POS, v.REF, alt
                )

                if feat_dict is None:
                    buf.append([v.CHROM, v.POS, v.REF, alt, ".", ".", "NO_DEPTH"])
                    n_skip += 1
                    continue

                # Build feature vector in EXACT training order
                feat_vec = np.array(
                    [feat_dict[fname] for fname in feature_names],
                    dtype=np.float64
                )

                raw = model.predict(feat_vec.reshape(1, -1))[0]
                prob = raw_to_prob(platt, raw)
                qual = round(prob_to_qual(prob), 2)
                verdict = "PASS" if prob >= tau else "DIRA_FILTERED"

                if prob >= tau:
                    n_pass += 1
                else:
                    n_filter += 1

                buf.append([v.CHROM, v.POS, v.REF, alt,
                            f"{prob:.6f}", str(qual), verdict])

            except Exception as e:
                buf.append([v.CHROM, v.POS, v.REF, alt, ".", ".", f"ERR:{e}"])
                n_skip += 1

        if len(buf) >= FLUSH:
            flush()

    flush()
    bam.close()
    ref.close()

    return tmp_path, n_pass, n_filter, n_skip


def _predict_worker(chrom, vcf_path, bam_path, ref_path,
                    model_path, platt_path, feature_names, tau, tmp_dir):
    """Picklable wrapper for Pool.map."""
    log.info(f"  [{chrom}] started")
    result = predict_chromosome(
        chrom, vcf_path, bam_path, ref_path,
        model_path, platt_path, feature_names, tau, tmp_dir
    )
    log.info(f"  [{chrom}] done — PASS={result[1]} FILTERED={result[2]} SKIP={result[3]}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# VCF writer — annotates original VCF with predictions
# ══════════════════════════════════════════════════════════════════════════════

def build_prediction_index(tmp_dir, chroms):
    """
    Read per-chromosome TSVs into a dict keyed by (chrom, pos, ref, alt)
    → (prob, qual, verdict).
    """
    index = {}
    for chrom in chroms:
        tmp_path = os.path.join(tmp_dir, f"{chrom}.tsv")
        if not os.path.exists(tmp_path):
            continue
        with open(tmp_path) as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                ch, pos, r, a, prob, qual, verdict = row
                index[(ch, int(pos), r, a)] = (prob, qual, verdict)
    return index


def write_annotated_vcf(vcf_path, pred_index, out_vcf, chroms):
    """
    Re-read the input VCF and annotate each record with DIRA predictions.
    """
    import cyvcf2

    vcf_in = cyvcf2.VCF(vcf_path)
    vcf_in.add_info_to_header({
        "ID": "DIRA_PROB",
        "Number": "A",
        "Type": "Float",
        "Description": "DIRA calibrated probability P(true indel)",
    })
    vcf_in.add_filter_to_header({
        "ID": "DIRA_FILTERED",
        "Description": "Filtered by DIRA ML classifier",
    })

    writer = cyvcf2.Writer(out_vcf, vcf_in)

    chrom_set = set(chroms)

    for v in vcf_in:
        # Only annotate chromosomes we processed
        if v.CHROM not in chrom_set:
            writer.write_record(v)
            continue

        probs = []
        any_filtered = False
        best_prob = None

        for alt in v.ALT:
            key = (v.CHROM, v.POS, v.REF, alt)
            pred = pred_index.get(key)

            if pred is None:
                # Variant not in our predictions (shouldn't happen, but safe)
                probs.append(None)
                continue

            prob_str, qual_str, verdict = pred

            if prob_str == ".":
                # SNP or skipped — pass through
                probs.append(None)
                continue

            p = float(prob_str)
            probs.append(p)

            if verdict == "DIRA_FILTERED":
                any_filtered = True

            if best_prob is None or p > best_prob:
                best_prob = p

        # Annotate
        valid_probs = [p for p in probs if p is not None]

        if valid_probs:
            v.INFO["DIRA_PROB"] = ",".join(f"{p:.4f}" for p in valid_probs)
            v.QUAL = round(prob_to_qual(best_prob), 2)

            # If ALL indel ALTs are filtered, mark the record as filtered
            indel_verdicts = []
            for alt in v.ALT:
                if len(v.REF) != len(alt):
                    pred = pred_index.get((v.CHROM, v.POS, v.REF, alt))
                    if pred and pred[2] not in ("SNP", ".", "NO_DEPTH"):
                        indel_verdicts.append(pred[2])

            if indel_verdicts and all(vd == "DIRA_FILTERED" for vd in indel_verdicts):
                v.FILTER = "DIRA_FILTERED"

        writer.write_record(v)

    writer.close()
    vcf_in.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run(candidate_vcf, bam_path, ref_path, model_dir, out_vcf, chroms, workers):

    model, platt, tau, feature_names = load_model(model_dir)

    d = Path(model_dir)
    model_path = str(d / "lgbm_model.txt")
    platt_path = str(d / "platt_scaler.pkl")

    n_workers = workers if workers > 0 else min(len(chroms), cpu_count())
    log.info(f"Processing {len(chroms)} chromosomes with {n_workers} workers")
    log.info(f"Threshold τ = {tau:.4f}")

    tmp_dir = tempfile.mkdtemp(prefix="dira_predict_")

    try:
        worker_fn = partial(
            _predict_worker,
            vcf_path=candidate_vcf,
            bam_path=bam_path,
            ref_path=ref_path,
            model_path=model_path,
            platt_path=platt_path,
            feature_names=feature_names,
            tau=tau,
            tmp_dir=tmp_dir,
        )

        with Pool(processes=n_workers) as pool:
            results = pool.map(worker_fn, chroms)

        # Aggregate counts
        total_pass = sum(r[1] for r in results)
        total_filter = sum(r[2] for r in results)
        total_skip = sum(r[3] for r in results)

        log.info(f"Prediction complete — PASS={total_pass:,}  "
                 f"FILTERED={total_filter:,}  SKIPPED={total_skip:,}")

        # Build index and write annotated VCF
        log.info("Writing annotated VCF...")
        pred_index = build_prediction_index(tmp_dir, chroms)
        write_annotated_vcf(candidate_vcf, pred_index, out_vcf, chroms)

        log.info(f"Output → {out_vcf}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DIRA — Parallelized inference v3")
    parser.add_argument("--candidate", required=True,
                        help="Candidate VCF (bgzipped + tabix-indexed)")
    parser.add_argument("--bam", required=True,
                        help="Indexed BAM file")
    parser.add_argument("--ref", required=True,
                        help="Reference FASTA (indexed)")
    parser.add_argument("--model", required=True,
                        help="Model directory (from train_dira_model_v3.py)")
    parser.add_argument("--out", required=True,
                        help="Output annotated VCF path")
    parser.add_argument("--chroms", nargs="+",
                        default=[f"chr{i}" for i in range(1, 19)],
                        help="Chromosomes to process (default: chr1-chr18)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (default: auto)")
    args = parser.parse_args()

    run(
        candidate_vcf=args.candidate,
        bam_path=args.bam,
        ref_path=args.ref,
        model_dir=args.model,
        out_vcf=args.out,
        chroms=args.chroms,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
