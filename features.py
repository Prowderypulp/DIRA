"""
DIRA — Feature Extraction
=========================
Computes per-indel feature vectors by re-querying the BAM at each locus.
All features are deterministic; no assembly or graph construction is performed.

Usage:
    extractor = DIRAFeatureExtractor(bam_path, ref_path)
    features  = extractor.extract(chrom, pos, ref_allele, alt_allele)
"""

import math
import numpy as np
import pysam
from dataclasses import dataclass, field
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

MIN_DEPTH       = 1          # avoid division by zero
BASE_AROUND_INDEL = 5        # bp window for BQ profile
NEARBY_VAR_WINDOW = 50       # bp window for nearby variant density


# ── Feature names (order must match extraction order below) ──────────────────

FEATURE_NAMES = [
    # Group 1 — Alignment Signal
    "allele_balance",
    "total_depth",
    "alt_depth",

    # Group 2 — Mapping Quality
    "mq_mean",
    "mq_var",
    "mq_mean_ref",          # MQ of ref-supporting reads (contrast signal)

    # Group 3 — Strand Bias
    "strand_ratio",         # alt forward / alt total
    "strand_ratio_ref",     # ref forward / ref total

    # Group 4 — Read Position Bias
    "read_pos_mean",
    "read_pos_var",

    # Group 5 — Soft-clip signal
    "softclip_rate_alt",    # fraction of alt reads with any soft-clip
    "softclip_mean_alt",    # mean soft-clip length on alt reads

    # Group 6 — Base Quality
    "bq_mean_alt",
    "bq_var_alt",
    "bq_mean_ref",
    "bq_drop_alt",          # mean BQ at indel pos vs flanking window

    # Group 7 — Indel Properties
    "indel_length",
    "is_insertion",         # 1 = insertion, 0 = deletion
    "inserted_seq_complexity",  # 0 for deletions; Shannon entropy of inserted bases

    # Group 8 — Reference Context
    "homopolymer_length",
    "gc_content_50bp",
    "tandem_repeat_score",  # crude: longest exact kmer repeat in 20bp window

    # Group 9 — Mappability proxy
    "low_mq_fraction",      # fraction of all reads at locus with MQ < 20

    # Group 10 — Nearby variant density
    "nearby_variant_count", # filled externally from VCF; default 0 here
]

N_FEATURES = len(FEATURE_NAMES)


# ── Helper functions ─────────────────────────────────────────────────────────

def _safe_mean(vals):
    return float(np.mean(vals)) if vals else 0.0

def _safe_var(vals):
    return float(np.var(vals)) if len(vals) > 1 else 0.0

def _shannon_entropy(seq: str) -> float:
    """Shannon entropy of a DNA sequence (bits). Returns 0 for empty/uniform."""
    if not seq:
        return 0.0
    seq = seq.upper()
    counts = {b: seq.count(b) for b in "ACGT"}
    n = len(seq)
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / n
            entropy -= p * math.log2(p)
    return entropy

def _homopolymer_length(ref_seq: str, pos: int) -> int:
    """Length of homopolymer run overlapping or adjacent to pos in ref_seq."""
    if not ref_seq or pos >= len(ref_seq):
        return 1
    base = ref_seq[pos].upper()
    if base not in "ACGT":
        return 1
    # expand left
    left = pos
    while left > 0 and ref_seq[left - 1].upper() == base:
        left -= 1
    # expand right
    right = pos
    while right < len(ref_seq) - 1 and ref_seq[right + 1].upper() == base:
        right += 1
    return right - left + 1

def _tandem_repeat_score(ref_seq: str, window: int = 20) -> int:
    """
    Crude tandem repeat score: length of the longest exact k-mer (k=2..6)
    that repeats consecutively within a window of the reference.
    """
    seq = ref_seq[:window].upper()
    best = 1
    for k in range(2, 7):
        for start in range(len(seq) - k):
            unit = seq[start:start + k]
            count = 1
            pos = start + k
            while pos + k <= len(seq) and seq[pos:pos + k] == unit:
                count += 1
                pos += k
            if count > 1:
                best = max(best, k * count)
    return best

def _gc_content(ref_seq: str) -> float:
    seq = ref_seq.upper()
    if not seq:
        return 0.5
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


# ── Main extractor ───────────────────────────────────────────────────────────

class DIRAFeatureExtractor:
    """
    Extract DIRA features for a single candidate indel.

    Parameters
    ----------
    bam_path : str
        Path to indexed BAM file.
    ref_path : str
        Path to indexed FASTA reference (e.g. GRCh38).
    min_base_quality : int
        Minimum base quality for reads to be considered (default 10).
    min_mapping_quality : int
        Minimum mapping quality for reads to be considered (default 0;
        we deliberately include low-MQ reads so we can measure their rate).
    """

    def __init__(
        self,
        bam_path: "/data/bam/HG002.GRCh38.2x250.bam",
        ref_path: "/data/reference/GRCh38.fa",
        min_base_quality: int = 10,
        min_mapping_quality: int = 0,
    ):
        self.bam = pysam.AlignmentFile(bam_path, "rb")
        self.ref = pysam.FastaFile(ref_path)
        self.min_bq  = min_base_quality
        self.min_mq  = min_mapping_quality

    def close(self):
        self.bam.close()
        self.ref.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Public API ───────────────────────────────────────────────────────────

    def extract(
        self,
        chrom: str,
        pos: int,           # 1-based VCF position
        ref_allele: str,
        alt_allele: str,
        nearby_variant_count: int = 0,
    ) -> np.ndarray:
        """
        Compute the full feature vector for one candidate indel.

        Returns
        -------
        np.ndarray of shape (N_FEATURES,), dtype float32.
        NaN-free: missing values are filled with 0.0.
        """
        bam_pos = pos - 1  # convert to 0-based for pysam

        # Fetch reference context (50 bp window centred on locus)
        ref_start = max(0, bam_pos - 25)
        ref_end   = bam_pos + 25
        try:
            ref_context = self.ref.fetch(chrom, ref_start, ref_end)
        except (ValueError, KeyError):
            ref_context = ""

        local_offset = bam_pos - ref_start  # position of indel within ref_context

        # Indel properties
        indel_length      = abs(len(ref_allele) - len(alt_allele))
        is_insertion      = int(len(alt_allele) > len(ref_allele))
        inserted_seq      = alt_allele[len(ref_allele):] if is_insertion else ""
        insert_complexity = _shannon_entropy(inserted_seq)

        # Reference context features (computed once)
        homopolymer_len   = _homopolymer_length(ref_context, local_offset)
        gc_content        = _gc_content(ref_context)
        tandem_score      = _tandem_repeat_score(ref_context[max(0, local_offset - 10): local_offset + 20])

        # Collect per-read statistics
        alt_mq, ref_mq                     = [], []
        alt_fwd, alt_rev, ref_fwd, ref_rev = 0, 0, 0, 0
        alt_read_pos                        = []
        alt_bq, ref_bq                      = [], []
        alt_softclip_lens                   = []
        all_mq_low                          = 0
        total_reads                         = 0

        # Determine the event string we expect in CIGAR / query for alt reads.
        # Simple heuristic: alt reads contain the inserted/deleted bases near locus.
        alt_len_diff = len(alt_allele) - len(ref_allele)

        for read in self.bam.fetch(chrom, bam_pos, bam_pos + 1):
            if read.is_unmapped or read.is_duplicate or read.is_secondary:
                continue
            total_reads += 1
            mq = read.mapping_quality or 0

            if mq < 20:
                all_mq_low += 1

            # Classify read as alt or ref supporting via simple overlap heuristic.
            # A production version should use pileup engine or allele matching.
            is_alt_read = _read_supports_indel(read, bam_pos, alt_len_diff)

            # Collect base quality at locus
            bq_at_locus = _base_quality_at(read, bam_pos)

            # Collect soft-clip info
            sc_len = _soft_clip_length(read)

            if is_alt_read:
                alt_mq.append(mq)
                if read.is_forward:
                    alt_fwd += 1
                else:
                    alt_rev += 1
                rp = _normalized_read_position(read, bam_pos)
                if rp is not None:
                    alt_read_pos.append(rp)
                if bq_at_locus is not None:
                    alt_bq.append(bq_at_locus)
                alt_softclip_lens.append(sc_len)
            else:
                ref_mq.append(mq)
                if read.is_forward:
                    ref_fwd += 1
                else:
                    ref_rev += 1
                if bq_at_locus is not None:
                    ref_bq.append(bq_at_locus)

        # ── Aggregate ────────────────────────────────────────────────────────

        alt_depth   = len(alt_mq)
        ref_depth   = len(ref_mq)
        total_depth = max(alt_depth + ref_depth, MIN_DEPTH)

        allele_balance = alt_depth / total_depth

        mq_mean     = _safe_mean(alt_mq)
        mq_var      = _safe_var(alt_mq)
        mq_mean_ref = _safe_mean(ref_mq)

        alt_total   = max(alt_fwd + alt_rev, MIN_DEPTH)
        ref_total   = max(ref_fwd + ref_rev, MIN_DEPTH)
        strand_ratio     = alt_fwd / alt_total
        strand_ratio_ref = ref_fwd / ref_total

        read_pos_mean = _safe_mean(alt_read_pos)
        read_pos_var  = _safe_var(alt_read_pos)

        softclip_rate_alt = (
            sum(1 for s in alt_softclip_lens if s > 0) / max(len(alt_softclip_lens), 1)
        )
        softclip_mean_alt = _safe_mean(alt_softclip_lens)

        bq_mean_alt = _safe_mean(alt_bq)
        bq_var_alt  = _safe_var(alt_bq)
        bq_mean_ref = _safe_mean(ref_bq)

        # BQ drop: mean BQ at locus vs flanking window (rough proxy)
        flanking_bq = _flanking_bq(self.bam, chrom, bam_pos, window=BASE_AROUND_INDEL)
        bq_drop_alt = _safe_mean(flanking_bq) - bq_mean_alt if flanking_bq else 0.0

        low_mq_fraction = all_mq_low / max(total_reads, 1)

        # ── Assemble vector ──────────────────────────────────────────────────

        vec = np.array([
            allele_balance,
            float(total_depth),
            float(alt_depth),
            mq_mean,
            mq_var,
            mq_mean_ref,
            strand_ratio,
            strand_ratio_ref,
            read_pos_mean,
            read_pos_var,
            softclip_rate_alt,
            softclip_mean_alt,
            bq_mean_alt,
            bq_var_alt,
            bq_mean_ref,
            bq_drop_alt,
            float(indel_length),
            float(is_insertion),
            insert_complexity,
            float(homopolymer_len),
            gc_content,
            float(tandem_score),
            low_mq_fraction,
            float(nearby_variant_count),
        ], dtype=np.float32)

        # Safety: replace any NaN/Inf with 0
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        assert len(vec) == N_FEATURES, f"Feature count mismatch: {len(vec)} vs {N_FEATURES}"
        return vec


# ── Read-level helpers ───────────────────────────────────────────────────────

def _read_supports_indel(
    read: pysam.AlignedSegment,
    bam_pos: int,
    alt_len_diff: int,
) -> bool:
    """
    Heuristic: a read supports the indel if its CIGAR contains an I or D
    operation that overlaps bam_pos and has the expected length sign.
    """
    if read.cigartuples is None:
        return False
    ref_pos = read.reference_start
    for op, length in read.cigartuples:
        # op codes: 0=M, 1=I, 2=D, 3=N, 4=S, 5=H, 6=P, 7=EQ, 8=X
        if op in (1, 2):  # insertion or deletion
            if abs(ref_pos - bam_pos) <= 2:  # within 2bp of called position
                if op == 1 and alt_len_diff > 0:
                    return True
                if op == 2 and alt_len_diff < 0:
                    return True
        if op not in (1, 4, 5):  # ops that consume reference
            ref_pos += length
    return False


def _normalized_read_position(
    read: pysam.AlignedSegment,
    bam_pos: int,
) -> Optional[float]:
    """Position of the indel within the read, normalised to [0, 1]."""
    if read.query_length is None or read.query_length == 0:
        return None
    # approximate query position from reference offset
    ref_offset = bam_pos - read.reference_start
    if ref_offset < 0:
        return None
    return min(ref_offset / read.query_length, 1.0)


def _base_quality_at(
    read: pysam.AlignedSegment,
    bam_pos: int,
) -> Optional[float]:
    """Base quality of the base at bam_pos in this read, or None."""
    if read.query_qualities is None:
        return None
    pairs = read.get_aligned_pairs(matches_only=True)
    for qpos, rpos in pairs:
        if rpos == bam_pos and qpos is not None:
            return float(read.query_qualities[qpos])
    return None


def _soft_clip_length(read: pysam.AlignedSegment) -> int:
    """Total soft-clip bases on this read."""
    if read.cigartuples is None:
        return 0
    return sum(length for op, length in read.cigartuples if op == 4)


def _flanking_bq(
    bam: pysam.AlignmentFile,
    chrom: str,
    bam_pos: int,
    window: int = 5,
) -> list:
    """
    Collect base qualities in a window around the indel position
    (excluding the indel position itself) for BQ-drop computation.
    """
    bqs = []
    start = max(0, bam_pos - window)
    end   = bam_pos + window + 1
    for read in bam.fetch(chrom, start, end):
        if read.is_unmapped or read.is_duplicate:
            continue
        for qpos, rpos in read.get_aligned_pairs(matches_only=True):
            if rpos is not None and start <= rpos < end and rpos != bam_pos:
                if read.query_qualities is not None and qpos is not None:
                    bqs.append(float(read.query_qualities[qpos]))
    return bqs
