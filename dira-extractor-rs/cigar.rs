//! CIGAR walking, indel-support classification, and aligned-pairs reconstruction.
//!
//! pysam's `get_aligned_pairs(matches_only=True)` semantics:
//! Walk the CIGAR. For each M/=/X (op codes 0/7/8) operation of length L,
//! emit L (qpos, rpos) pairs incrementing both query and reference positions.
//! Skip I (1) for reference advance, skip D/N (2/3) for query advance,
//! skip S/H (4/5) entirely.

use rust_htslib::bam::record::{Cigar, CigarStringView, Record};

/// Aligned (qpos, rpos) pairs for matched positions only.
/// Used for NBQ, pileup base lookup, read-end distance, flanking BQ.
pub struct AlignedPairs {
    pub pairs: Vec<(usize, i64)>,           // (qpos, rpos), sorted by rpos ascending
}

impl AlignedPairs {
    pub fn from_record(rec: &Record) -> Self {
        let mut pairs = Vec::with_capacity(rec.seq_len());
        let mut qpos: usize = 0;
        let mut rpos: i64 = rec.pos();
        for c in rec.cigar().iter() {
            match *c {
                Cigar::Match(l) | Cigar::Equal(l) | Cigar::Diff(l) => {
                    for _ in 0..l {
                        pairs.push((qpos, rpos));
                        qpos += 1;
                        rpos += 1;
                    }
                }
                Cigar::Ins(l) | Cigar::SoftClip(l) => {
                    qpos += l as usize;
                }
                Cigar::Del(l) | Cigar::RefSkip(l) => {
                    rpos += l as i64;
                }
                Cigar::HardClip(_) | Cigar::Pad(_) => { /* no-op */ }
            }
        }
        AlignedPairs { pairs }
    }

    /// O(log n) lookup of qpos at a given rpos via binary search.
    pub fn qpos_at(&self, target_rpos: i64) -> Option<usize> {
        match self.pairs.binary_search_by_key(&target_rpos, |&(_, r)| r) {
            Ok(idx) => Some(self.pairs[idx].0),
            Err(_) => None,
        }
    }
}

/// Returns true if the read contains an I or D CIGAR op near var_pos_0based
/// matching the candidate indel direction and (approximately) length.
/// Replicates v6's `read_supports_indel`.
pub fn read_supports_indel(
    cigar: &CigarStringView,
    read_start: i64,
    var_pos_0based: i64,
    is_insertion: bool,
    indel_length_abs: i64,
) -> bool {
    let tolerance: i64 = 2;
    let mut ref_pos: i64 = read_start;
    for c in cigar.iter() {
        match *c {
            Cigar::Match(l) | Cigar::Equal(l) | Cigar::Diff(l) => {
                ref_pos += l as i64;
            }
            Cigar::Ins(l) => {
                if is_insertion {
                    if (ref_pos - var_pos_0based).abs() <= tolerance {
                        let len_diff = (l as i64 - indel_length_abs).abs();
                        let max_diff = std::cmp::max(1, indel_length_abs / 2);
                        if len_diff <= max_diff {
                            return true;
                        }
                    }
                }
                // Insertion does not advance reference.
            }
            Cigar::Del(l) => {
                if !is_insertion {
                    let del_start = ref_pos;
                    let del_end = ref_pos + l as i64;
                    if del_start - tolerance <= var_pos_0based
                        && var_pos_0based <= del_end + tolerance
                    {
                        let len_diff = (l as i64 - indel_length_abs).abs();
                        let max_diff = std::cmp::max(1, indel_length_abs / 2);
                        if len_diff <= max_diff {
                            return true;
                        }
                    }
                }
                ref_pos += l as i64;
            }
            Cigar::RefSkip(l) => {
                ref_pos += l as i64;
            }
            Cigar::SoftClip(_) | Cigar::HardClip(_) | Cigar::Pad(_) => { /* no-op */ }
        }
    }
    false
}

/// CIGAR signature near a variant site for haplotype consistency features.
/// Matches v6's `get_cigar_signature_at_site` semantics.
pub fn cigar_signature(
    cigar: &CigarStringView,
    read_start: i64,
    var_pos_0based: i64,
    window: i64,
) -> Vec<(u8, u32)> {
    let mut sig: Vec<(u8, u32)> = Vec::new();
    let mut ref_pos: i64 = read_start;
    for c in cigar.iter() {
        match *c {
            Cigar::Match(l) | Cigar::Equal(l) | Cigar::Diff(l) => {
                let block_end = ref_pos + l as i64;
                if block_end >= var_pos_0based - window
                    && ref_pos <= var_pos_0based + window
                {
                    let op_code: u8 = match *c {
                        Cigar::Match(_) => 0,
                        Cigar::Equal(_) => 7,
                        Cigar::Diff(_) => 8,
                        _ => unreachable!(),
                    };
                    sig.push((op_code, std::cmp::min(l, 10)));
                }
                ref_pos += l as i64;
            }
            Cigar::Ins(l) => {
                if (ref_pos - var_pos_0based).abs() <= window {
                    sig.push((1, l));
                }
            }
            Cigar::Del(l) => {
                if ref_pos - window <= var_pos_0based
                    && var_pos_0based <= ref_pos + l as i64 + window
                {
                    sig.push((2, l));
                }
                ref_pos += l as i64;
            }
            Cigar::RefSkip(l) => {
                ref_pos += l as i64;
            }
            Cigar::SoftClip(l) => {
                sig.push((4, std::cmp::min(l, 20)));
            }
            Cigar::HardClip(_) | Cigar::Pad(_) => { /* no-op */ }
        }
    }
    sig
}

/// Total soft-clip length across the read.
pub fn softclip_total(cigar: &CigarStringView) -> u32 {
    cigar.iter().map(|c| match *c {
        Cigar::SoftClip(l) => l,
        _ => 0,
    }).sum()
}
