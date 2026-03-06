#!/usr/bin/env python3

import pandas as pd
from cyvcf2 import VCF
import argparse


def load_truth(truth_vcf):

    truth = set()

    vcf = VCF(truth_vcf)

    for v in vcf:

        if len(v.REF) == len(v.ALT[0]):
            continue

        truth.add((v.CHROM, v.POS, v.REF, v.ALT[0]))

    return truth


def build_dataset(feature_csv, truth_vcf, out_file):

    print("Loading feature table...")
    df = pd.read_csv(feature_csv)

    print("Loading truth set...")
    truth = load_truth(truth_vcf)

    print("Assigning labels...")

    labels = []

    for row in df.itertuples(index=False):

        key = (row.chrom, row.pos, row.ref, row.alt)

        if key in truth:
            labels.append(1)
        else:
            labels.append(0)

    df["label"] = labels

    print("Saving parquet dataset...")
    df.to_parquet(out_file, index=False)

    print("Dataset shape:", df.shape)
    print(df.label.value_counts())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--features", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    build_dataset(args.features, args.truth, args.out)
