#!/usr/bin/env python3

import pysam
import numpy as np
import math
from collections import Counter
from cyvcf2 import VCF
import csv


def shannon_entropy(arr):
    if len(arr) == 0:
        return 0
    counts = Counter(arr)
    total = len(arr)
    ent = 0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def compute_nbq(read, var_pos, window=5):
    """
    Correct NBQ using reference-aligned coordinates.
    """
    quals = read.query_qualities
    total = 0
    count = 0

    for qpos, rpos in read.get_aligned_pairs(matches_only=True):

        if rpos is None:
            continue

        if abs(rpos - var_pos) <= window:

            total += quals[qpos]
            count += 1

    if count == 0:
        return 0

    return total / count


def extract_features(vcf_path, bam_path, ref_path, out_csv):

    vcf = VCF(vcf_path)
    bam = pysam.AlignmentFile(bam_path)
    ref = pysam.FastaFile(ref_path)

    header = [
        "chrom","pos","ref","alt",
        "allele_balance","total_depth","alt_depth",
        "mq_mean_alt","mq_var_alt","mq_mean_ref",
        "strand_ratio_alt","strand_ratio_ref",
        "read_end_dist_mean","read_end_dist_var",
        "softclip_rate_alt","softclip_mean_alt",
        "bq_mean_alt","bq_var_alt","bq_mean_ref",
        "nbq_alt","nbq_ref",
        "bq_drop_alt",
        "indel_length","is_insertion",
        "insert_entropy",
        "homopolymer_length","gc_content_50bp",
        "udp_dp_ratio","pileup_entropy"
    ]

    with open(out_csv, "w") as f:

        writer = csv.writer(f)
        writer.writerow(header)

        for var in vcf:

            if len(var.REF) == len(var.ALT[0]):
                continue

            chrom = var.CHROM
            pos = var.POS
            alt = var.ALT[0]

            reads = []
            bases = []

            alt_reads = []
            ref_reads = []

            for col in bam.pileup(chrom, pos-1, pos, truncate=True):

                if col.pos != pos-1:
                    continue

                for pr in col.pileups:

                    if pr.is_del or pr.is_refskip:
                        continue

                    r = pr.alignment
                    qpos = pr.query_position
                    base = r.query_sequence[qpos]

                    reads.append((r, qpos))
                    bases.append(base)

                    if base == alt[0]:
                        alt_reads.append((r, qpos))
                    else:
                        ref_reads.append((r, qpos))

            depth = len(reads)
            alt_depth = len(alt_reads)

            if depth == 0:
                continue

            allele_balance = alt_depth / depth

            mq_alt = [r.mapping_quality for r,_ in alt_reads]
            mq_ref = [r.mapping_quality for r,_ in ref_reads]

            mq_mean_alt = np.mean(mq_alt) if mq_alt else 0
            mq_var_alt = np.var(mq_alt) if mq_alt else 0
            mq_mean_ref = np.mean(mq_ref) if mq_ref else 0

            strand_alt = sum(not r.is_reverse for r,_ in alt_reads)
            strand_ratio_alt = strand_alt / alt_depth if alt_depth else 0

            strand_ref = sum(not r.is_reverse for r,_ in ref_reads)
            strand_ratio_ref = strand_ref / len(ref_reads) if ref_reads else 0

            read_end_dist = []
            nbq_alt = []
            nbq_ref = []

            for r, qpos in reads:

                read_len = r.query_length
                read_end_dist.append(min(qpos, read_len-qpos))

                nbq = compute_nbq(r, pos)

                if (r,qpos) in alt_reads:
                    nbq_alt.append(nbq)
                else:
                    nbq_ref.append(nbq)

            read_end_dist_mean = np.mean(read_end_dist)
            read_end_dist_var = np.var(read_end_dist)

            nbq_alt = np.mean(nbq_alt) if nbq_alt else 0
            nbq_ref = np.mean(nbq_ref) if nbq_ref else 0

            bq_alt = [np.mean(r.query_qualities) for r,_ in alt_reads]
            bq_ref = [np.mean(r.query_qualities) for r,_ in ref_reads]

            bq_mean_alt = np.mean(bq_alt) if bq_alt else 0
            bq_var_alt = np.var(bq_alt) if bq_alt else 0
            bq_mean_ref = np.mean(bq_ref) if bq_ref else 0

            bq_drop_alt = bq_mean_ref - bq_mean_alt

            softclips = []

            for r,_ in alt_reads:
                if r.cigartuples:
                    sc = sum(l for op,l in r.cigartuples if op==4)
                else:
                    sc = 0
                softclips.append(sc)

            softclip_rate_alt = np.mean(softclips) if softclips else 0
            softclip_mean_alt = softclip_rate_alt

            indel_length = len(alt) - len(var.REF)
            is_insertion = 1 if indel_length > 0 else 0

            insert_entropy = shannon_entropy(alt) if is_insertion else 0

            seq = ref.fetch(chrom, pos-50, pos+50)

            gc = (seq.count("G")+seq.count("C")) / len(seq)

            center = seq[50]
            homopoly = 1

            i = 49
            while i >= 0 and seq[i] == center:
                homopoly += 1
                i -= 1

            i = 51
            while i < len(seq) and seq[i] == center:
                homopoly += 1
                i += 1

            udp = len(reads)
            dp = sum(r.mapping_quality >= 20 for r,_ in reads)

            udp_dp = udp / dp if dp > 0 else 0

            pile_entropy = shannon_entropy(bases)

            writer.writerow([
                chrom,pos,var.REF,alt,
                allele_balance,depth,alt_depth,
                mq_mean_alt,mq_var_alt,mq_mean_ref,
                strand_ratio_alt,strand_ratio_ref,
                read_end_dist_mean,read_end_dist_var,
                softclip_rate_alt,softclip_mean_alt,
                bq_mean_alt,bq_var_alt,bq_mean_ref,
                nbq_alt,nbq_ref,
                bq_drop_alt,
                indel_length,is_insertion,
                insert_entropy,
                homopoly,gc,
                udp_dp,pile_entropy
            ])


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--vcf", required=True)
    parser.add_argument("--bam", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    extract_features(args.vcf, args.bam, args.ref, args.out)
