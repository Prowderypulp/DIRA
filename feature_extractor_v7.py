#!/usr/bin/env python3
"""
DIRA — Feature Extractor v7
==============================
Thin Python wrapper around the Rust extractor binary. Preserves the v5/v6
CLI so the rest of the DIRA pipeline (build_dataset_v3, predict_dira_v3,
train_v2) is unaffected. All extraction work is done by the Rust binary;
this wrapper only marshals arguments and validates the output.

Binary discovery order:
    1. --binary command-line argument (if provided)
    2. $DIRA_EXTRACTOR environment variable
    3. dira-extractor on $PATH
    4. ~/DIRA/dira-extractor-rs/target/release/dira-extractor

Usage (drop-in replacement for feature_extractor_v5/v6):
    python feature_extractor_v7.py \\
        --vcf     candidates.vcf.gz \\
        --bam     sample.bam \\
        --ref     GRCh38.fa \\
        --out     features.csv \\
        [--chroms chr1 chr2 ...] \\
        [--workers 8]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_CHROMS = [f"chr{i}" for i in range(1, 23)]

FALLBACK_BINARY = Path.home() / "DIRA" / "dira-extractor-rs" / "target" / "release" / "dira-extractor"


def locate_binary(explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).is_file():
            sys.exit(f"[v7] --binary path does not exist: {explicit}")
        return explicit
    env = os.environ.get("DIRA_EXTRACTOR")
    if env and Path(env).is_file():
        return env
    on_path = shutil.which("dira-extractor")
    if on_path:
        return on_path
    if FALLBACK_BINARY.is_file():
        return str(FALLBACK_BINARY)
    sys.exit(
        "[v7] Could not locate dira-extractor binary. Set $DIRA_EXTRACTOR, "
        "add it to PATH, or pass --binary."
    )


def main():
    p = argparse.ArgumentParser(
        description="DIRA feature extractor v7 (Rust-backed)")
    p.add_argument("--vcf", required=True, help="Candidate indel VCF (bgzipped)")
    p.add_argument("--bam", required=True, help="Aligned reads BAM (indexed)")
    p.add_argument("--ref", required=True, help="Reference FASTA (indexed)")
    p.add_argument("--out", required=True, help="Output features CSV")
    p.add_argument("--chroms", nargs="+", default=DEFAULT_CHROMS,
                   help="Chromosomes to process (default: chr1..chr22)")
    p.add_argument("--workers", type=int, default=0,
                   help="Worker threads (0 = auto / all cores)")
    p.add_argument("--chunk-size", type=int, default=1_000_000,
                   help="Region chunk size in bp (default: 1,000,000)")
    p.add_argument("--binary", default=None,
                   help="Path to dira-extractor binary (overrides auto-detection)")
    args = p.parse_args()

    for flag, path in (("--vcf", args.vcf), ("--bam", args.bam), ("--ref", args.ref)):
        if not Path(path).is_file():
            sys.exit(f"[v7] {flag} not found: {path}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    binary = locate_binary(args.binary)

    cmd = [
        binary,
        "--vcf", args.vcf,
        "--bam", args.bam,
        "--ref", args.ref,
        "--out", args.out,
        "--workers", str(args.workers),
        "--chunk-size", str(args.chunk_size),
        "--chroms", *args.chroms,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[v7] dira-extractor exited with code {result.returncode}")

    if not Path(args.out).is_file():
        sys.exit(f"[v7] extractor returned success but output file missing: {args.out}")


if __name__ == "__main__":
    main()
