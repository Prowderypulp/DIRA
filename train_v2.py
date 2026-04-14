#!/usr/bin/env python3
"""
DIRA — LightGBM Training Script v3
====================================
Trains an indel filtering model on features from feature_extractor_v3.

Changes from v2:
  1. Pileup-derived depth features (allele_balance, total_depth, alt_depth)
     kept as features — these are NOT caller leakage, they come from your
     own BAM pileup, not VCF INFO fields.
  2. softclip_mean_alt dropped — it's identical to softclip_rate_alt.
  3. Stratified train/val/test splits (70/15/15) with clear ratios.
  4. Platt calibration fitted on held-out val set and saved for inference.
  5. Model saved as lgbm_model.txt (matches predict script expectation).
  6. Chromosome-aware split option to avoid positional leakage.
  7. Full results JSON with all metrics + threshold + feature list.
  8. Reproducible: all random seeds pinned.

Usage:
    python train_dira_model_v3.py --data train_features.parquet --out model_dir/
    python train_dira_model_v3.py --data train_features.parquet --out model_dir/ --split-by-chrom
"""

import argparse
import json
import os
import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Columns that are NOT features (identifiers + label)
# ──────────────────────────────────────────────────────────────────────────────

ID_COLS = ["chrom", "pos", "ref", "alt", "label"]

# Features to explicitly drop (redundant / problematic)
DROP_FEATURES = [
    "softclip_mean_alt",        # identical to softclip_rate_alt
]


# ──────────────────────────────────────────────────────────────────────────────
# Chromosome-aware splitting (avoids positional leakage from nearby variants)
# ──────────────────────────────────────────────────────────────────────────────

def chrom_split(df, val_chroms=None, test_chroms=None):
    """
    Split by chromosome to avoid spatial leakage.
    Defaults: test=chr17,chr18  val=chr15,chr16  train=rest
    """
    if test_chroms is None:
        test_chroms = {"chr20"}
    if val_chroms is None:
        val_chroms = {"chr19"}

    mask_test = df["chrom"].isin(test_chroms)
    mask_val = df["chrom"].isin(val_chroms)
    mask_train = ~(mask_test | mask_val)

    return df[mask_train], df[mask_val], df[mask_test]


# ──────────────────────────────────────────────────────────────────────────────
# Platt calibration
# ──────────────────────────────────────────────────────────────────────────────

def fit_platt_scaler(raw_scores, y_true):
    """
    Fit a logistic regression on raw LightGBM scores to produce
    calibrated probabilities. Returns the fitted sklearn estimator.
    """
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(raw_scores.reshape(-1, 1), y_true)
    return lr


def platt_predict(scaler, raw_scores):
    """Apply Platt scaler to get calibrated probabilities."""
    return scaler.predict_proba(raw_scores.reshape(-1, 1))[:, 1]


# ──────────────────────────────────────────────────────────────────────────────
# Optimal threshold via PR curve
# ──────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true, probs):
    """
    Find threshold that maximizes F1 on the precision-recall curve.
    Returns (tau, precision, recall, f1).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)

    best_idx = np.argmax(f1)
    return (
        float(thresholds[best_idx]),
        float(precision[best_idx]),
        float(recall[best_idx]),
        float(f1[best_idx]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, probs, tau, split_name="Test"):
    """Compute and print all metrics for a split."""
    preds = (probs >= tau).astype(int)

    p = precision_score(y_true, preds, zero_division=0)
    r = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    brier = brier_score_loss(y_true, probs)
    auroc = roc_auc_score(y_true, probs)
    auprc = average_precision_score(y_true, probs)

    print(f"\n{'=' * 50}")
    print(f"  {split_name} Results  (τ = {tau:.4f})")
    print(f"{'=' * 50}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUROC     : {auroc:.4f}")
    print(f"  AUPRC     : {auprc:.4f}")
    print(f"  Brier     : {brier:.4f}")
    print(f"  N         : {len(y_true):,}  (pos={y_true.sum():,}  neg={(1 - y_true).sum():,.0f})")
    print(f"{'=' * 50}")

    return dict(
        precision=round(p, 4),
        recall=round(r, 4),
        f1=round(f1, 4),
        auroc=round(auroc, 4),
        auprc=round(auprc, 4),
        brier=round(brier, 4),
        n=int(len(y_true)),
        n_pos=int(y_true.sum()),
        n_neg=int((1 - y_true).sum()),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────────────────────────────────────

def train(data_path, out_dir, split_by_chrom=False):

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading dataset...")
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"Dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    # ── Feature selection ─────────────────────────────────────────────────────
    feature_cols = [
        c for c in df.columns
        if c not in ID_COLS and c not in DROP_FEATURES
    ]

    print(f"Features ({len(feature_cols)}):")
    for i, f in enumerate(feature_cols):
        print(f"  {i + 1:2d}. {f}")
    print()

    # ── Split ─────────────────────────────────────────────────────────────────
    if split_by_chrom:
        print("Splitting by chromosome (train=chr1-14, val=chr15-16, test=chr17-18)")
        df_train, df_val, df_test = chrom_split(df)
    else:
        print("Splitting randomly (70/15/15, stratified)")
        df_train, df_temp = train_test_split(
            df, test_size=0.30, random_state=SEED, stratify=df["label"]
        )
        df_val, df_test = train_test_split(
            df_temp, test_size=0.50, random_state=SEED, stratify=df_temp["label"]
        )

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    X_val = df_val[feature_cols].values
    y_val = df_val["label"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values

    print(f"Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    # ── Class imbalance ───────────────────────────────────────────────────────
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = neg / max(pos, 1)

    print(f"Train pos={pos:,}  neg={neg:,}  scale_pos_weight={scale_pos_weight:.3f}\n")

    # ── LightGBM training ─────────────────────────────────────────────────────
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_data_in_leaf=50,
        scale_pos_weight=scale_pos_weight,
        seed=SEED,
        verbosity=-1,
    )

    print("Training LightGBM...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ],
    )
    print(f"Best iteration: {model.best_iteration}")

    # ── Raw scores ────────────────────────────────────────────────────────────
    raw_val = model.predict(X_val, num_iteration=model.best_iteration)
    raw_test = model.predict(X_test, num_iteration=model.best_iteration)

    # ── Platt calibration on validation set ───────────────────────────────────
    print("\nFitting Platt calibration on validation set...")
    platt = fit_platt_scaler(raw_val, y_val)

    cal_val = platt_predict(platt, raw_val)
    cal_test = platt_predict(platt, raw_test)

    brier_raw = brier_score_loss(y_val, raw_val)
    brier_cal = brier_score_loss(y_val, cal_val)
    print(f"Brier (val) — raw: {brier_raw:.4f}  calibrated: {brier_cal:.4f}")

    # ── Optimal threshold on calibrated val probs ─────────────────────────────
    tau, val_p, val_r, val_f1 = find_optimal_threshold(y_val, cal_val)
    print(f"\nOptimal τ = {tau:.4f}  (val P={val_p:.4f} R={val_r:.4f} F1={val_f1:.4f})")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    val_metrics = evaluate(y_val, cal_val, tau, split_name="Validation")
    test_metrics = evaluate(y_test, cal_test, tau, split_name="Test")

    # ── Feature importance ────────────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)

    print("\nTop features (by gain):")
    print(importance.head(10).to_string(index=False))

    # ── Save artifacts ────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    # Model
    model_path = os.path.join(out_dir, "lgbm_model.txt")
    model.save_model(model_path, num_iteration=model.best_iteration)

    # Platt scaler
    platt_path = os.path.join(out_dir, "platt_scaler.pkl")
    with open(platt_path, "wb") as f:
        pickle.dump(platt, f)

    # Feature importance
    importance.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    # Features list (for inference validation)
    with open(os.path.join(out_dir, "features.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Full results
    results = dict(
        threshold=tau,
        scale_pos_weight=round(scale_pos_weight, 4),
        best_iteration=model.best_iteration,
        n_features=len(feature_cols),
        features=feature_cols,
        split_method="chromosome" if split_by_chrom else "random_stratified",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        params=params,
    )

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Threshold (backward compat with predict script)
    with open(os.path.join(out_dir, "threshold.json"), "w") as f:
        json.dump({"tau": tau}, f)

    print(f"\nAll artifacts saved to {out_dir}/")
    print(f"  lgbm_model.txt        — LightGBM model")
    print(f"  platt_scaler.pkl      — Platt calibration scaler")
    print(f"  features.json         — ordered feature list")
    print(f"  feature_importance.csv — gain + split importance")
    print(f"  results.json          — full metrics + params + threshold")
    print(f"  threshold.json        — threshold τ only")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DIRA — Train LightGBM indel filter")
    parser.add_argument("--data", required=True,
                        help="Labelled dataset (.parquet or .csv)")
    parser.add_argument("--out", required=True,
                        help="Output directory for model artifacts")
    parser.add_argument("--split-by-chrom", action="store_true",
                        help="Split by chromosome instead of random "
                             "(train=chr1-14, val=chr15-16, test=chr17-18)")
    args = parser.parse_args()

    train(args.data, args.out, split_by_chrom=args.split_by_chrom)


if __name__ == "__main__":
    main()
