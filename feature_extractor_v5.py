#!/usr/bin/env python3
"""
Feature Extractor v5 — DNAscope-Inspired + Read Consistency
=============================================================
Builds on v4 (CIGAR-aware classification). New features inspired by
DNAscope (Sentieon) and DeepVariant papers:

  DNAscope-inspired:
    - Haplotype consistency / local assembly entropy approximation
    - ReadPosEndDist on alt vs ref reads separately
    - Refined mismatch rate using actual reference comparison

  DeepVariant-inspired:
    - CIGAR agreement among alt reads (do they all show same indel?)
    - Fragment-level support (both mates support indel?)

  Additional:
    - Insert size deviation (alt vs ref)
    - Flanking base quality (5bp each side of indel)
    - Tandem repeat length at variant site
    - MQ0 fraction (reads with MQ=0)
    - Clipping difference (alt vs ref softclip rates)

Target: GCP L2 (n2-standard-8).
"""

import pysam
import math
import os
import sys
import csv
import argparse
import tempfile
import shutil
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial


# ---------------------------------------------------------------------------
# Pure-python stats
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

def _median(arr):
    if not arr:
        return 0.0
    s = sorted(arr)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

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
# NBQ with early exit
# ---------------------------------------------------------------------------

def compute_nbq(read, var_pos, window=5):
    quals = read.query_qualities
    if quals is None:
        return 0.0
    total = count = 0
    lo, hi = var_pos - window, var_pos + window
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


# ---------------------------------------------------------------------------
# CIGAR-aware indel support detection (same as v4)
# ---------------------------------------------------------------------------

def read_supports_indel(read, var_pos_0based, is_insertion, indel_length):
    if read.cigartuples is None:
        return False

    tolerance = 2
    ref_pos = read.reference_start
    target_op = 1 if is_insertion else 2

    for op, length in read.cigartuples:
        if op == 0 or op == 7 or op == 8:
            ref_pos += length
        elif op == 1:
            if op == target_op:
                if abs(ref_pos - var_pos_0based) <= tolerance:
                    if abs(length - abs(indel_length)) <= max(1, abs(indel_length) // 2):
                        return True
        elif op == 2:
            if op == target_op:
                del_start = ref_pos
                del_end = ref_pos + length
                if del_start - tolerance <= var_pos_0based <= del_end + tolerance:
                    if abs(length - abs(indel_length)) <= max(1, abs(indel_length) // 2):
                        return True
            ref_pos += length
        elif op == 3:
            ref_pos += length
        elif op == 4 or op == 5:
            pass

    return False


# ---------------------------------------------------------------------------
# Extract CIGAR signature at variant site (for agreement computation)
# ---------------------------------------------------------------------------

def get_cigar_signature_at_site(read, var_pos_0based, window=3):
    """
    Extract a tuple describing the CIGAR operations near the variant.
    Used to check whether alt reads agree on the same indel structure.
    """
    if read.cigartuples is None:
        return ()

    sig = []
    ref_pos = read.reference_start

    for op, length in read.cigartuples:
        if op == 0 or op == 7 or op == 8:
            # Check if this M/=/X block overlaps the window
            block_end = ref_pos + length
            if block_end >= var_pos_0based - window and ref_pos <= var_pos_0based + window:
                sig.append((op, min(length, 10)))  # cap length to reduce noise
            ref_pos += length
        elif op == 1:
            if abs(ref_pos - var_pos_0based) <= window:
                sig.append((1, length))
        elif op == 2:
            if ref_pos - window <= var_pos_0based <= ref_pos + length + window:
                sig.append((2, length))
            ref_pos += length
        elif op == 3:
            ref_pos += length
        elif op == 4:
            sig.append((4, min(length, 20)))
        # op == 5: hard clip, skip

    return tuple(sig)


# ---------------------------------------------------------------------------
# Tandem repeat detection
# ---------------------------------------------------------------------------

def find_tandem_repeat_length(seq, pos_in_seq, max_unit=6):
    """
    Find the longest tandem repeat spanning pos_in_seq.
    Returns (unit_length, total_repeat_length).
    """
    if not seq or pos_in_seq < 0 or pos_in_seq >= len(seq):
        return 1, 1

    best_total = 1
    best_unit = 1

    for unit_len in range(1, max_unit + 1):
        if pos_in_seq + unit_len > len(seq):
            break
        unit = seq[pos_in_seq:pos_in_seq + unit_len]

        # Extend left
        left = pos_in_seq
        while left >= unit_len and seq[left - unit_len:left] == unit:
            left -= unit_len

        # Extend right
        right = pos_in_seq + unit_len
        while right + unit_len <= len(seq) and seq[right:right + unit_len] == unit:
            right += unit_len

        total = right - left
        if total > best_total:
            best_total = total
            best_unit = unit_len

    return best_unit, best_total


# ---------------------------------------------------------------------------
# Flanking base quality
# ---------------------------------------------------------------------------

def flanking_bq(read, var_pos, ref_obj, chrom, window=5):
    """
    Compute mean BQ in the flanking ±window around the variant,
    and count actual mismatches vs reference.
    Returns (mean_flank_bq, mismatch_count, aligned_count).
    """
    quals = read.query_qualities
    query = read.query_sequence
    if quals is None or query is None:
        return 0.0, 0, 0

    lo, hi = var_pos - window, var_pos + window
    bq_sum = 0
    bq_count = 0
    mismatches = 0
    aligned = 0

    try:
        ref_seq = ref_obj.fetch(chrom, max(0, lo), hi + 1)
        ref_offset = max(0, lo)
    except Exception:
        ref_seq = None
        ref_offset = 0

    for qpos, rpos in read.get_aligned_pairs(matches_only=True):
        if rpos is None:
            continue
        if rpos < lo:
            continue
        if rpos > hi:
            break

        bq_sum += quals[qpos]
        bq_count += 1
        aligned += 1

        # Actual mismatch detection
        if ref_seq is not None:
            ref_idx = rpos - ref_offset
            if 0 <= ref_idx < len(ref_seq):
                if query[qpos].upper() != ref_seq[ref_idx].upper():
                    mismatches += 1

    mean_bq = bq_sum / bq_count if bq_count else 0.0
    return mean_bq, mismatches, aligned


# ---------------------------------------------------------------------------
# CSV header
# ---------------------------------------------------------------------------

HEADER = [
    "chrom", "pos", "ref", "alt",

    # ── v3 core features ──────────────────────────────────────────────────
    "allele_balance", "total_depth", "alt_depth",
    "mq_mean_alt", "mq_var_alt", "mq_mean_ref",
    "strand_ratio_alt", "strand_ratio_ref",
    "read_end_dist_mean", "read_end_dist_var",
    "softclip_rate_alt",
    "bq_mean_alt", "bq_var_alt", "bq_mean_ref",
    "nbq_alt", "nbq_ref",
    "bq_drop_alt",
    "indel_length", "is_insertion", "insert_entropy",
    "homopolymer_length", "gc_content_50bp",
    "udp_dp_ratio",
    "pileup_entropy",

    # ── v4 features ───────────────────────────────────────────────────────
    "nm_mean_alt", "nm_mean_ref",
    "proper_pair_rate_alt", "proper_pair_rate_ref",
    "mismatch_rate_alt",

    # ── v5 NEW: DNAscope-inspired ─────────────────────────────────────────
    "cigar_agreement",          # fraction of alt reads with identical CIGAR signature
    "haplotype_entropy",        # Shannon entropy of CIGAR signatures (assembly proxy)
    "fragment_support_rate",    # fraction of alt reads whose mate also supports indel

    # ── v5 NEW: read quality stratification ───────────────────────────────
    "read_end_dist_mean_alt",   # read-end distance for alt reads only
    "read_end_dist_mean_ref",   # read-end distance for ref reads only
    "softclip_rate_ref",        # softclip rate in ref reads (for comparison)
    "softclip_diff",            # softclip_rate_alt - softclip_rate_ref

    # ── v5 NEW: insert size ───────────────────────────────────────────────
    "isize_mean_alt",           # mean abs insert size for alt reads
    "isize_mean_ref",           # mean abs insert size for ref reads
    "isize_dev",                # isize_mean_alt - isize_mean_ref

    # ── v5 NEW: flanking quality + true mismatch ──────────────────────────
    "flank_bq_alt",             # mean BQ in ±5bp flanking region (alt reads)
    "flank_bq_ref",             # mean BQ in ±5bp flanking region (ref reads)
    "true_mismatch_rate_alt",   # actual mismatches vs reference near site (alt)

    # ── v5 NEW: tandem repeat context ─────────────────────────────────────
    "tandem_repeat_unit",       # repeat unit length at site
    "tandem_repeat_length",     # total repeat tract length

    # ── v5 NEW: mapping quality extras ────────────────────────────────────
    "mq0_fraction",             # fraction of reads with MQ=0
    "mq_diff",                  # mq_mean_alt - mq_mean_ref
]


# ---------------------------------------------------------------------------
# Worker: process all indels on one chromosome
# ---------------------------------------------------------------------------

def process_chromosome(chrom, vcf_path, bam_path, ref_path, tmp_dir):
    from cyvcf2 import VCF

    vcf = VCF(vcf_path)
    bam = pysam.AlignmentFile(bam_path, "rb")
    ref = pysam.FastaFile(ref_path)

    tmp_path = os.path.join(tmp_dir, f"{chrom}.csv")
    rows_buf = []
    FLUSH_EVERY = 500

    def flush():
        nonlocal rows_buf
        if not rows_buf:
            return
        with open(tmp_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(rows_buf)
        rows_buf = []

    try:
        var_iter = vcf(chrom)
    except Exception:
        bam.close()
        ref.close()
        return tmp_path

    # Pre-index mate read names for fragment support
    # (too expensive genome-wide; we check per-site via mate lookup)

    for var in var_iter:
        if var.ALT is None or len(var.ALT) == 0:
            continue
        if len(var.REF) == len(var.ALT[0]):
            continue

        pos = var.POS
        alt = var.ALT[0]
        ref_allele = var.REF
        pos0 = pos - 1

        indel_length = len(alt) - len(ref_allele)
        is_insertion = 1 if indel_length > 0 else 0

        # ------------------------------------------------------------------
        # Collect reads via fetch, classify with CIGAR
        # ------------------------------------------------------------------
        all_reads = []
        alt_reads = []
        ref_reads = []
        alt_set = set()
        alt_read_names = set()
        bases = []

        for read in bam.fetch(chrom, max(0, pos0 - 1), pos0 + 2):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.is_duplicate:
                continue
            if read.reference_start > pos0 or read.reference_end is None:
                continue
            if read.reference_end <= pos0:
                continue

            idx = len(all_reads)
            all_reads.append(read)

            if read_supports_indel(read, pos0, is_insertion == 1, indel_length):
                alt_reads.append(read)
                alt_set.add(idx)
                alt_read_names.add(read.query_name)
            else:
                ref_reads.append(read)

            for qpos, rpos in read.get_aligned_pairs():
                if rpos == pos0 and qpos is not None:
                    bases.append(read.query_sequence[qpos])
                    break

        depth = len(all_reads)
        alt_depth = len(alt_reads)
        if depth == 0:
            continue

        allele_balance = alt_depth / depth

        # ------------------------------------------------------------------
        # Mapping quality
        # ------------------------------------------------------------------
        mq_alt = [r.mapping_quality for r in alt_reads]
        mq_ref = [r.mapping_quality for r in ref_reads]
        mq_mean_alt = _mean(mq_alt)
        mq_var_alt = _var(mq_alt)
        mq_mean_ref = _mean(mq_ref)

        # ------------------------------------------------------------------
        # Strand ratios
        # ------------------------------------------------------------------
        strand_alt = sum(1 for r in alt_reads if not r.is_reverse)
        strand_ratio_alt = strand_alt / alt_depth if alt_depth else 0.0
        strand_ref = sum(1 for r in ref_reads if not r.is_reverse)
        strand_ratio_ref = strand_ref / len(ref_reads) if ref_reads else 0.0

        # ------------------------------------------------------------------
        # Read-end distance & NBQ (all reads + stratified)
        # ------------------------------------------------------------------
        read_end_dist_all = []
        read_end_dist_alt_vals = []
        read_end_dist_ref_vals = []
        nbq_alt_vals = []
        nbq_ref_vals = []

        for i, r in enumerate(all_reads):
            qpos_at_site = None
            for qp, rp in r.get_aligned_pairs():
                if rp == pos0:
                    qpos_at_site = qp
                    break

            rlen = r.query_length or r.infer_read_length() or 150
            if qpos_at_site is not None:
                d = min(qpos_at_site, rlen - qpos_at_site)
            else:
                d = rlen // 2

            read_end_dist_all.append(d)

            nbq = compute_nbq(r, pos)
            if i in alt_set:
                nbq_alt_vals.append(nbq)
                read_end_dist_alt_vals.append(d)
            else:
                nbq_ref_vals.append(nbq)
                read_end_dist_ref_vals.append(d)

        read_end_dist_mean = _mean(read_end_dist_all)
        read_end_dist_var = _var(read_end_dist_all)
        read_end_dist_mean_alt = _mean(read_end_dist_alt_vals)
        read_end_dist_mean_ref = _mean(read_end_dist_ref_vals)

        nbq_alt_mean = _mean(nbq_alt_vals)
        nbq_ref_mean = _mean(nbq_ref_vals)

        # ------------------------------------------------------------------
        # Base quality (whole-read average)
        # ------------------------------------------------------------------
        def _read_mean_bq(r):
            q = r.query_qualities
            if q is None or len(q) == 0:
                return 0.0
            return sum(q) / len(q)

        bq_alt = [_read_mean_bq(r) for r in alt_reads]
        bq_ref = [_read_mean_bq(r) for r in ref_reads]
        bq_mean_alt = _mean(bq_alt)
        bq_var_alt = _var(bq_alt)
        bq_mean_ref = _mean(bq_ref)
        bq_drop_alt = bq_mean_ref - bq_mean_alt

        # ------------------------------------------------------------------
        # Soft-clip stats (alt AND ref for comparison)
        # ------------------------------------------------------------------
        def _softclip_total(r):
            ct = r.cigartuples
            return sum(l for op, l in ct if op == 4) if ct else 0

        sc_alt = [_softclip_total(r) for r in alt_reads]
        sc_ref = [_softclip_total(r) for r in ref_reads]
        softclip_rate_alt = _mean(sc_alt)
        softclip_rate_ref = _mean(sc_ref)
        softclip_diff = softclip_rate_alt - softclip_rate_ref

        # ------------------------------------------------------------------
        # Indel properties
        # ------------------------------------------------------------------
        insert_entropy = shannon_entropy(alt) if is_insertion else 0.0

        # ------------------------------------------------------------------
        # Reference context (homopolymer + GC + tandem repeat)
        # ------------------------------------------------------------------
        ctx_start = max(0, pos0 - 50)
        ctx_end = pos0 + 50
        seq = ref.fetch(chrom, ctx_start, ctx_end)

        if len(seq) == 0:
            gc, homopoly = 0.0, 1
            tr_unit, tr_length = 1, 1
        else:
            gc = (seq.count("G") + seq.count("C")) / len(seq)
            ci = pos0 - ctx_start

            if ci < 0 or ci >= len(seq):
                homopoly = 1
                tr_unit, tr_length = 1, 1
            else:
                center = seq[ci]
                homopoly = 1
                j = ci - 1
                while j >= 0 and seq[j] == center:
                    homopoly += 1; j -= 1
                j = ci + 1
                while j < len(seq) and seq[j] == center:
                    homopoly += 1; j += 1

                tr_unit, tr_length = find_tandem_repeat_length(seq, ci)

        # ------------------------------------------------------------------
        # UDP / DP ratio + MQ0
        # ------------------------------------------------------------------
        dp = sum(1 for r in all_reads if r.mapping_quality >= 20)
        udp_dp = depth / dp if dp > 0 else 0.0

        mq0_count = sum(1 for r in all_reads if r.mapping_quality == 0)
        mq0_fraction = mq0_count / depth

        # ------------------------------------------------------------------
        # Pileup entropy
        # ------------------------------------------------------------------
        pile_entropy = shannon_entropy(bases)

        # ------------------------------------------------------------------
        # NM (edit distance) stats
        # ------------------------------------------------------------------
        def _get_nm(r):
            try:
                return r.get_tag("NM")
            except KeyError:
                return 0

        nm_alt = [_get_nm(r) for r in alt_reads]
        nm_ref = [_get_nm(r) for r in ref_reads]
        nm_mean_alt = _mean(nm_alt)
        nm_mean_ref = _mean(nm_ref)

        # ------------------------------------------------------------------
        # Proper pair rates
        # ------------------------------------------------------------------
        pp_alt = sum(1 for r in alt_reads if r.is_proper_pair)
        proper_pair_rate_alt = pp_alt / alt_depth if alt_depth else 0.0
        pp_ref = sum(1 for r in ref_reads if r.is_proper_pair)
        proper_pair_rate_ref = pp_ref / len(ref_reads) if ref_reads else 0.0

        # ------------------------------------------------------------------
        # v5 NEW: CIGAR agreement among alt reads
        # ------------------------------------------------------------------
        if alt_reads:
            sigs = [get_cigar_signature_at_site(r, pos0) for r in alt_reads]
            sig_counts = Counter(sigs)
            most_common_count = sig_counts.most_common(1)[0][1]
            cigar_agreement = most_common_count / len(sigs)
            haplotype_entropy = shannon_entropy(sigs)
        else:
            cigar_agreement = 0.0
            haplotype_entropy = 0.0

        # ------------------------------------------------------------------
        # v5 NEW: Fragment-level support
        # ------------------------------------------------------------------
        if alt_reads:
            mate_support = 0
            for r in alt_reads:
                if r.is_paired and not r.mate_is_unmapped:
                    mate_name = r.query_name
                    # Check if mate is also in alt reads
                    # (We already collected alt_read_names)
                    # A read's mate has the same query_name
                    # Count pairs where both reads are in alt_read_names
                    # Each pair will be counted twice, so we use a set
                    mate_support += 1 if mate_name in alt_read_names else 0
            # mate_support counts each alt read whose mate is also alt
            # Divide by alt_depth to get rate
            fragment_support_rate = mate_support / alt_depth
        else:
            fragment_support_rate = 0.0

        # ------------------------------------------------------------------
        # v5 NEW: Insert size features
        # ------------------------------------------------------------------
        def _abs_isize(r):
            return abs(r.template_length) if r.template_length != 0 else 0

        isize_alt = [_abs_isize(r) for r in alt_reads if r.is_proper_pair and r.template_length != 0]
        isize_ref = [_abs_isize(r) for r in ref_reads if r.is_proper_pair and r.template_length != 0]
        isize_mean_alt = _mean(isize_alt)
        isize_mean_ref = _mean(isize_ref)
        isize_dev = isize_mean_alt - isize_mean_ref

        # ------------------------------------------------------------------
        # v5 NEW: Flanking BQ + true mismatch rate (actual ref comparison)
        # ------------------------------------------------------------------
        flank_bq_alt_vals = []
        flank_mm_alt_counts = []
        flank_aligned_alt_counts = []
        for r in alt_reads:
            fbq, mm, al = flanking_bq(r, pos0, ref, chrom, window=5)
            flank_bq_alt_vals.append(fbq)
            flank_mm_alt_counts.append(mm)
            flank_aligned_alt_counts.append(al)

        flank_bq_ref_vals = []
        for r in ref_reads:
            fbq, _, _ = flanking_bq(r, pos0, ref, chrom, window=5)
            flank_bq_ref_vals.append(fbq)

        flank_bq_alt = _mean(flank_bq_alt_vals)
        flank_bq_ref = _mean(flank_bq_ref_vals)

        total_mm = sum(flank_mm_alt_counts)
        total_al = sum(flank_aligned_alt_counts)
        true_mismatch_rate_alt = total_mm / total_al if total_al > 0 else 0.0

        # v4 compat: mismatch_rate_alt (now using true mismatches)
        mismatch_rate_alt = true_mismatch_rate_alt

        # ------------------------------------------------------------------
        # MQ diff
        # ------------------------------------------------------------------
        mq_diff = mq_mean_alt - mq_mean_ref

        # ------------------------------------------------------------------
        # Append row
        # ------------------------------------------------------------------
        rows_buf.append([
            chrom, pos, ref_allele, alt,

            # v3 core
            allele_balance, depth, alt_depth,
            mq_mean_alt, mq_var_alt, mq_mean_ref,
            strand_ratio_alt, strand_ratio_ref,
            read_end_dist_mean, read_end_dist_var,
            softclip_rate_alt,
            bq_mean_alt, bq_var_alt, bq_mean_ref,
            nbq_alt_mean, nbq_ref_mean,
            bq_drop_alt,
            indel_length, is_insertion, insert_entropy,
            homopoly, gc,
            udp_dp,
            pile_entropy,

            # v4
            nm_mean_alt, nm_mean_ref,
            proper_pair_rate_alt, proper_pair_rate_ref,
            mismatch_rate_alt,

            # v5 new
            cigar_agreement, haplotype_entropy, fragment_support_rate,
            read_end_dist_mean_alt, read_end_dist_mean_ref,
            softclip_rate_ref, softclip_diff,
            isize_mean_alt, isize_mean_ref, isize_dev,
            flank_bq_alt, flank_bq_ref, true_mismatch_rate_alt,
            tr_unit, tr_length,
            mq0_fraction, mq_diff,
        ])

        if len(rows_buf) >= FLUSH_EVERY:
            flush()

    flush()
    bam.close()
    ref.close()
    return tmp_path


# ---------------------------------------------------------------------------
# Merge + Main
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


def _worker_wrapper(chrom, vcf_path, bam_path, ref_path, tmp_dir):
    print(f"  [{chrom}] started", file=sys.stderr, flush=True)
    result = process_chromosome(chrom, vcf_path, bam_path, ref_path, tmp_dir)
    print(f"  [{chrom}] done", file=sys.stderr, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="DIRA Feature Extractor v5")
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--bam", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chroms", nargs="+",
                        default=[f"chr{i}" for i in range(1, 19)],
                        help="Chromosomes to process (default: chr1-chr18)")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    chroms = args.chroms
    n_workers = args.workers if args.workers > 0 else min(len(chroms), cpu_count())

    print(f"[feature_extractor_v5] {len(chroms)} chromosomes, "
          f"{n_workers} workers", file=sys.stderr)

    tmp_dir = tempfile.mkdtemp(prefix="feat_v5_")
    try:
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
        print(f"[feature_extractor_v5] Done → {args.out}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
