#!/usr/bin/env python3
"""
DIRA — Predict v3
==================
Features already extracted. Model already trained.
Just score and annotate the VCF.

Input:  features CSV + candidate VCF + model dir
Output: annotated VCF with PASS/DIRA_FILTERED

Usage:
    python predict_dira_v3.py \
        --features  chr20_features.csv \
        --vcf       HG002_chr20_indels.norm.vcf.gz \
        --model     model_v3/ \
        --out       HG002_chr20_dira.vcf
"""

import argparse
import json
import logging
import math
import pickle
from pathlib import Path

import cyvcf2
import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dira.predict")


def load_model(model_dir):
    d = Path(model_dir)

    model = lgb.Booster(model_file=str(d / "lgbm_model.txt"))

    with open(d / "features.json") as f:
        feature_names = json.load(f)

    platt = None
    if (d / "platt_scaler.pkl").exists():
        with open(d / "platt_scaler.pkl", "rb") as f:
            platt = pickle.load(f)

    tau = None
    for fname in ("results.json", "threshold.json"):
        p = d / fname
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            tau = data.get("threshold") or data.get("tau")
            if tau:
                break
    tau = float(tau) if tau else 0.5

    log.info(f"Model: {model.num_feature()} features, τ={tau:.4f}")
    return model, platt, tau, feature_names


def raw_to_prob(platt, raw):
    if platt is not None:
        return float(platt.predict_proba([[raw]])[0][1])
    return 1.0 / (1.0 + math.exp(-raw))


def prob_to_qual(prob):
    return -10.0 * math.log10(1.0 - min(prob, 1.0 - 1e-9))


def run(features_path, vcf_path, model_dir, out_vcf):

    model, platt, tau, feature_names = load_model(model_dir)

    # ── Load features, score everything in one shot ───────────────────────
    log.info(f"Loading features: {features_path}")
    df = pd.read_csv(features_path)

    # Align column names between extractor versions
    # Model expects both read_pos_* and read_end_dist_* as separate features
    if "read_pos_mean" not in df.columns and "read_end_dist_mean" in df.columns:
        df["read_pos_mean"] = df["read_end_dist_mean"]
    if "read_pos_var" not in df.columns and "read_end_dist_var" in df.columns:
        df["read_pos_var"] = df["read_end_dist_var"]

    # Check all required features exist
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        log.error(f"Missing features in CSV: {missing}")
        log.error(f"CSV columns: {list(df.columns)}")
        raise KeyError(f"Missing features: {missing}")

    log.info(f"{len(df):,} indels loaded")

    X = df[feature_names].values.astype(np.float64)
    raw_scores = model.predict(X)
    probs = np.array([raw_to_prob(platt, r) for r in raw_scores])

    df["prob"] = probs
    log.info("Predictions done")

    # ── Build lookup: (chrom, pos, ref, alt) → prob ───────────────────────
    lookup = {}
    for i, row in df.iterrows():
        key = (str(row["chrom"]), int(row["pos"]), str(row["ref"]), str(row["alt"]))
        lookup[key] = row["prob"]

    # ── Walk VCF, annotate, write ─────────────────────────────────────────
    vcf_in = cyvcf2.VCF(vcf_path)
    vcf_in.add_info_to_header({
        "ID": "DIRA_PROB", "Number": "A", "Type": "Float",
        "Description": "DIRA probability P(true indel)",
    })
    vcf_in.add_filter_to_header({
        "ID": "DIRA_FILTERED",
        "Description": "Filtered by DIRA ML classifier",
    })
    writer = cyvcf2.Writer(out_vcf, vcf_in)

    n_pass = n_filter = n_miss = 0

    for v in vcf_in:
        indel_probs = []

        for alt in v.ALT:
            if len(v.REF) == len(alt):
                continue
            key = (v.CHROM, v.POS, v.REF, alt)
            prob = lookup.get(key)
            if prob is not None:
                indel_probs.append(prob)
            else:
                n_miss += 1

        if indel_probs:
            best = max(indel_probs)
            v.INFO["DIRA_PROB"] = ",".join(f"{p:.4f}" for p in indel_probs)
            v.QUAL = round(prob_to_qual(best), 2)

            if best >= tau:
                n_pass += 1
            else:
                v.FILTER = "DIRA_FILTERED"
                n_filter += 1

        writer.write_record(v)

    writer.close()
    vcf_in.close()

    log.info(f"PASS={n_pass:,}  FILTERED={n_filter:,}  NO_MATCH={n_miss:,}")
    log.info(f"Output → {out_vcf}")


def main():
    p = argparse.ArgumentParser(description="DIRA predict")
    p.add_argument("--features", required=True, help="Pre-extracted features CSV")
    p.add_argument("--vcf", required=True, help="Candidate VCF to annotate")
    p.add_argument("--model", required=True, help="Model directory")
    p.add_argument("--out", required=True, help="Output VCF")
    args = p.parse_args()
    run(args.features, args.vcf, args.model, args.out)


if __name__ == "__main__":
    main()
