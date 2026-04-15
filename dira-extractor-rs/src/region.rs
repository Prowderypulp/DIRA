//! Per-region worker — sequential pileup walk (v3).
//!
//! Instead of `bam.fetch(pos-1, pos+2)` per indel site (which does a BGZF seek
//! per site), we do ONE `bam.fetch(region)` and walk reads sequentially. A
//! sliding window buffer maintains currently-active reads. For each indel
//! site (in sorted order), we:
//!   - Evict reads whose end <= pos0
//!   - Pull new reads from the iterator whose start <= pos0
//!   - Compute features from the current buffer
//!
//! This turns N seeks into 1 seek per region.

use anyhow::{Context, Result};
use rust_htslib::bam::{IndexedReader as BamReader, Read as BamRead, Record};
use rust_htslib::bcf::{IndexedReader as BcfReader, Read as BcfRead};
use rust_htslib::faidx::Reader as FaReader;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::cigar::{cigar_signature, read_supports_indel, softclip_total, AlignedPairs};
use crate::features::FeatureRow;

pub struct Region {
    pub chrom: String,
    pub start: u64,
    pub end: u64,
}

/// Indel site awaiting feature computation.
struct IndelSite {
    pos: i64,
    pos0: i64,
    ref_allele: String,
    alt_allele: String,
    is_insertion: bool,
    indel_length: i64,
    indel_length_abs: i64,
}

/// A read + its precomputed aligned pairs, kept in the active window.
struct BufferedRead {
    record: Record,
    pairs: AlignedPairs,
    read_end: i64, // cigar.end_pos() — reference position 1-past-last aligned base
}

pub fn process_region(
    region: &Region,
    bam: &mut BamReader,
    vcf: &mut BcfReader,
    fa: &FaReader,
) -> Result<Vec<FeatureRow>> {
    // ── Step 1: collect all indel sites in this region (from VCF) ─────────
    let header = vcf.header().clone();
    let rid = match header.name2rid(region.chrom.as_bytes()) {
        Ok(r) => r,
        Err(_) => return Ok(vec![]),
    };
    vcf.fetch(rid, region.start - 1, Some(region.end))?;

    let mut sites: Vec<IndelSite> = Vec::new();
    let mut vrec = vcf.empty_record();
    while let Some(r) = vcf.read(&mut vrec) {
        r?;
        let pos = vrec.pos() + 1;
        if (pos as u64) < region.start || (pos as u64) > region.end {
            continue;
        }
        let alleles = vrec.alleles();
        if alleles.len() < 2 {
            continue;
        }
        let ref_a = std::str::from_utf8(alleles[0])?.to_string();
        let alt_a = std::str::from_utf8(alleles[1])?.to_string();
        if ref_a.len() == alt_a.len() {
            continue;
        }
        let indel_len = alt_a.len() as i64 - ref_a.len() as i64;
        sites.push(IndelSite {
            pos,
            pos0: pos - 1,
            ref_allele: ref_a,
            alt_allele: alt_a,
            is_insertion: indel_len > 0,
            indel_length: indel_len,
            indel_length_abs: indel_len.abs(),
        });
    }

    if sites.is_empty() {
        return Ok(vec![]);
    }

    // VCFs are position-sorted but double-check.
    sites.sort_by_key(|s| s.pos0);

    // ── Step 2: sequential BAM walk over the region, sliding window ────────
    let fetch_start = std::cmp::max(0i64, region.start as i64 - 500); // padding for reads spanning in
    let fetch_end = region.end as i64 + 500;
    bam.fetch((region.chrom.as_str(), fetch_start, fetch_end))
        .with_context(|| format!("bam fetch region {}:{}-{}", region.chrom, fetch_start, fetch_end))?;

    let mut buffer: VecDeque<BufferedRead> = VecDeque::with_capacity(1024);
    let mut next_read = Record::new();
    let mut next_read_ready = false;
    let mut bam_exhausted = false;

    // Helper: try to load the next usable read into next_read.
    // Returns true if next_read now holds a valid record, false if BAM exhausted.
    fn pull_next(
        bam: &mut BamReader,
        buf: &mut Record,
        exhausted: &mut bool,
    ) -> Result<bool> {
        loop {
            match bam.read(buf) {
                None => {
                    *exhausted = true;
                    return Ok(false);
                }
                Some(r) => {
                    r?;
                    if buf.is_unmapped() || buf.is_secondary()
                        || buf.is_supplementary() || buf.is_duplicate()
                    {
                        continue;
                    }
                    return Ok(true);
                }
            }
        }
    }

    // Prime the first read.
    if pull_next(bam, &mut next_read, &mut bam_exhausted)? {
        next_read_ready = true;
    }

    let mut rows: Vec<FeatureRow> = Vec::with_capacity(sites.len());

    for site in &sites {
        // (a) Admit reads whose start <= site.pos0 into the buffer.
        while next_read_ready && next_read.pos() <= site.pos0 {
            let cigar = next_read.cigar();
            let read_end = cigar.end_pos();
            // pysam requires read.reference_end > pos0 to be a candidate.
            // If read already ends before site, skip; later sites are further
            // right so it'd be useless.
            if read_end > site.pos0 {
                let pairs = AlignedPairs::from_record(&next_read);
                buffer.push_back(BufferedRead {
                    record: next_read.clone(),
                    pairs,
                    read_end,
                });
            }
            // Advance.
            if !pull_next(bam, &mut next_read, &mut bam_exhausted)? {
                next_read_ready = false;
                break;
            }
        }

        // (b) Evict reads that no longer overlap site.pos0.
        while let Some(front) = buffer.front() {
            if front.read_end <= site.pos0 {
                buffer.pop_front();
            } else {
                break;
            }
        }

        // (c) Also filter: read.reference_start > pos0 — skip from front scan.
        // With VCF sorted and reads position-sorted, any read in the buffer has
        // pos <= site.pos0 already (we only admitted those). But we still need
        // to exclude reads whose start is strictly > pos0 — but those can't be
        // in the buffer by construction. So the buffer IS the candidate set.

        // (d) Compute features for this site from the current buffer.
        if let Some(row) = compute_site_features(site, &region.chrom, &buffer, fa)? {
            rows.push(row);
        }
    }

    Ok(rows)
}

fn compute_site_features(
    site: &IndelSite,
    chrom: &str,
    buffer: &VecDeque<BufferedRead>,
    fa: &FaReader,
) -> Result<Option<FeatureRow>> {
    if buffer.is_empty() {
        return Ok(None);
    }

    // Classify alt vs ref.
    let mut alt_indices: HashSet<usize> = HashSet::new();
    let mut alt_qnames: HashSet<Vec<u8>> = HashSet::new();
    for (i, br) in buffer.iter().enumerate() {
        let cigar = br.record.cigar();
        if read_supports_indel(
            &cigar, br.record.pos(), site.pos0,
            site.is_insertion, site.indel_length_abs,
        ) {
            alt_indices.insert(i);
            alt_qnames.insert(br.record.qname().to_vec());
        }
    }

    let depth = buffer.len();
    let alt_depth = alt_indices.len();
    let ref_depth = depth - alt_depth;
    let allele_balance = alt_depth as f64 / depth as f64;

    // Pre-fetch flanking ref (once per site).
    let flank_lo = std::cmp::max(0, site.pos0 - 5) as usize;
    let flank_hi = (site.pos0 + 5) as usize;
    let flank_seq: Vec<u8> = fa
        .fetch_seq(chrom, flank_lo, flank_hi)
        .map(|s| s.to_ascii_uppercase())
        .unwrap_or_default();
    let flank_offset = flank_lo as i64;

    let ctx_start = std::cmp::max(0, site.pos0 - 50) as usize;
    let ctx_end = (site.pos0 + 50 - 1) as usize;
    let ctx_seq: Vec<u8> = fa
        .fetch_seq(chrom, ctx_start, ctx_end)
        .map(|s| s.to_ascii_uppercase())
        .unwrap_or_default();
    let (gc, homopoly, tr_unit, tr_length) = compute_context(&ctx_seq, site.pos0 - ctx_start as i64);

    let mut mq_alt: Vec<f64> = Vec::with_capacity(alt_depth);
    let mut mq_ref: Vec<f64> = Vec::with_capacity(ref_depth);
    let mut bq_alt: Vec<f64> = Vec::with_capacity(alt_depth);
    let mut bq_ref: Vec<f64> = Vec::with_capacity(ref_depth);
    let mut nm_alt: Vec<f64> = Vec::with_capacity(alt_depth);
    let mut nm_ref: Vec<f64> = Vec::with_capacity(ref_depth);
    let mut sc_alt: Vec<f64> = Vec::with_capacity(alt_depth);
    let mut sc_ref: Vec<f64> = Vec::with_capacity(ref_depth);
    let mut nbq_alt: Vec<f64> = Vec::with_capacity(alt_depth);
    let mut nbq_ref: Vec<f64> = Vec::with_capacity(ref_depth);
    let mut ed_all: Vec<f64> = Vec::with_capacity(depth);
    let mut ed_alt: Vec<f64> = Vec::with_capacity(alt_depth);
    let mut ed_ref: Vec<f64> = Vec::with_capacity(ref_depth);
    let mut isize_alt: Vec<f64> = Vec::with_capacity(alt_depth);
    let mut isize_ref: Vec<f64> = Vec::with_capacity(ref_depth);
    let mut flank_bq_alt_v: Vec<f64> = Vec::new();
    let mut flank_bq_ref_v: Vec<f64> = Vec::new();
    let mut flank_mm_total: u64 = 0;
    let mut flank_al_total: u64 = 0;

    let mut strand_alt_fwd = 0usize;
    let mut strand_ref_fwd = 0usize;
    let mut pp_alt = 0usize;
    let mut pp_ref = 0usize;
    let mut mq0_count = 0usize;
    let mut dp_mq20 = 0usize;
    let mut bases_at_site: Vec<u8> = Vec::with_capacity(depth);
    let mut alt_sigs: Vec<Vec<(u8, u32)>> = Vec::with_capacity(alt_depth);
    let mut frag_support = 0usize;

    for (i, br) in buffer.iter().enumerate() {
        let r = &br.record;
        let pairs = &br.pairs;

        let mq = r.mapq() as f64;
        if r.mapq() == 0 { mq0_count += 1; }
        if r.mapq() >= 20 { dp_mq20 += 1; }
        let quals = r.qual();
        let seq_len = r.seq_len();
        let bq_mean = if quals.is_empty() {
            0.0
        } else {
            quals.iter().map(|&q| q as f64).sum::<f64>() / quals.len() as f64
        };
        let nm = r.aux(b"NM").ok().and_then(|a| {
            use rust_htslib::bam::record::Aux;
            match a {
                Aux::U8(v) => Some(v as f64),
                Aux::U16(v) => Some(v as f64),
                Aux::U32(v) => Some(v as f64),
                Aux::I8(v) => Some(v as f64),
                Aux::I16(v) => Some(v as f64),
                Aux::I32(v) => Some(v as f64),
                _ => None,
            }
        }).unwrap_or(0.0);
        let sc = softclip_total(&r.cigar()) as f64;

        let qpos_at_site = pairs.qpos_at(site.pos0);
        // NBQ window uses 1-based pos to match Python v6 semantics (off-by-one).
        let nbq = compute_nbq(quals, &pairs.pairs, site.pos, 5);
        let rlen = if seq_len > 0 { seq_len } else { 150 };
        let ed = match qpos_at_site {
            Some(qp) => std::cmp::min(qp, rlen.saturating_sub(qp)) as f64,
            None => (rlen / 2) as f64,
        };
        ed_all.push(ed);

        if let Some(qp) = qpos_at_site {
            let seq = r.seq();
            if qp < seq_len {
                bases_at_site.push(seq[qp]);
            }
        }

        let (fbq, mm, al) = flanking_bq(
            quals, &r.seq().as_bytes(), &pairs.pairs,
            site.pos0, &flank_seq, flank_offset, 5,
        );

        let is_alt = alt_indices.contains(&i);
        if is_alt {
            mq_alt.push(mq); bq_alt.push(bq_mean); nm_alt.push(nm); sc_alt.push(sc);
            nbq_alt.push(nbq); ed_alt.push(ed);
            if !r.is_reverse() { strand_alt_fwd += 1; }
            if r.is_proper_pair() { pp_alt += 1; }
            if r.is_proper_pair() && r.insert_size() != 0 {
                isize_alt.push(r.insert_size().unsigned_abs() as f64);
            }
            flank_bq_alt_v.push(fbq);
            flank_mm_total += mm;
            flank_al_total += al;
            alt_sigs.push(cigar_signature(&r.cigar(), r.pos(), site.pos0, 3));
            if r.is_paired() && !r.is_mate_unmapped() && alt_qnames.contains(r.qname()) {
                frag_support += 1;
            }
        } else {
            mq_ref.push(mq); bq_ref.push(bq_mean); nm_ref.push(nm); sc_ref.push(sc);
            nbq_ref.push(nbq); ed_ref.push(ed);
            if !r.is_reverse() { strand_ref_fwd += 1; }
            if r.is_proper_pair() { pp_ref += 1; }
            if r.is_proper_pair() && r.insert_size() != 0 {
                isize_ref.push(r.insert_size().unsigned_abs() as f64);
            }
            flank_bq_ref_v.push(fbq);
        }
    }

    let mq_mean_alt = mean(&mq_alt);
    let mq_mean_ref = mean(&mq_ref);
    let mq_var_alt = var(&mq_alt);
    let bq_mean_alt = mean(&bq_alt);
    let bq_var_alt = var(&bq_alt);
    let bq_mean_ref = mean(&bq_ref);
    let bq_drop_alt = bq_mean_ref - bq_mean_alt;
    let nm_mean_alt = mean(&nm_alt);
    let nm_mean_ref = mean(&nm_ref);
    let softclip_rate_alt = mean(&sc_alt);
    let softclip_rate_ref = mean(&sc_ref);
    let softclip_diff = softclip_rate_alt - softclip_rate_ref;
    let nbq_alt_mean = mean(&nbq_alt);
    let nbq_ref_mean = mean(&nbq_ref);
    let read_end_dist_mean = mean(&ed_all);
    let read_end_dist_var = var(&ed_all);
    let read_end_dist_mean_alt = mean(&ed_alt);
    let read_end_dist_mean_ref = mean(&ed_ref);
    let isize_mean_alt = mean(&isize_alt);
    let isize_mean_ref = mean(&isize_ref);
    let isize_dev = isize_mean_alt - isize_mean_ref;
    let flank_bq_alt = mean(&flank_bq_alt_v);
    let flank_bq_ref = mean(&flank_bq_ref_v);
    let true_mismatch_rate_alt = if flank_al_total > 0 {
        flank_mm_total as f64 / flank_al_total as f64
    } else { 0.0 };
    let mq_diff = mq_mean_alt - mq_mean_ref;

    let strand_ratio_alt = if alt_depth > 0 { strand_alt_fwd as f64 / alt_depth as f64 } else { 0.0 };
    let strand_ratio_ref = if ref_depth > 0 { strand_ref_fwd as f64 / ref_depth as f64 } else { 0.0 };
    let proper_pair_rate_alt = if alt_depth > 0 { pp_alt as f64 / alt_depth as f64 } else { 0.0 };
    let proper_pair_rate_ref = if ref_depth > 0 { pp_ref as f64 / ref_depth as f64 } else { 0.0 };
    let udp_dp_ratio = if dp_mq20 > 0 { depth as f64 / dp_mq20 as f64 } else { 0.0 };
    let mq0_fraction = mq0_count as f64 / depth as f64;

    let pileup_entropy = entropy_bytes(&bases_at_site);
    let insert_entropy = if site.is_insertion {
        entropy_bytes(site.alt_allele.as_bytes())
    } else { 0.0 };

    let (cigar_agreement, haplotype_entropy) = if !alt_sigs.is_empty() {
        let mut counts: HashMap<&Vec<(u8, u32)>, usize> = HashMap::new();
        for s in &alt_sigs {
            *counts.entry(s).or_insert(0) += 1;
        }
        let max_count = counts.values().copied().max().unwrap_or(0);
        let agreement = max_count as f64 / alt_sigs.len() as f64;
        let total = alt_sigs.len() as f64;
        let ent: f64 = counts.values().map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        }).sum();
        (agreement, ent)
    } else { (0.0, 0.0) };

    let fragment_support_rate = if alt_depth > 0 {
        frag_support as f64 / alt_depth as f64
    } else { 0.0 };

    Ok(Some(FeatureRow {
        chrom: chrom.to_string(), pos: site.pos,
        ref_allele: site.ref_allele.clone(), alt_allele: site.alt_allele.clone(),
        allele_balance, total_depth: depth as i64, alt_depth: alt_depth as i64,
        mq_mean_alt, mq_var_alt, mq_mean_ref,
        strand_ratio_alt, strand_ratio_ref,
        read_end_dist_mean, read_end_dist_var, softclip_rate_alt,
        bq_mean_alt, bq_var_alt, bq_mean_ref, nbq_alt: nbq_alt_mean, nbq_ref: nbq_ref_mean,
        bq_drop_alt, indel_length: site.indel_length, is_insertion: site.is_insertion as i64, insert_entropy,
        homopolymer_length: homopoly, gc_content_50bp: gc, udp_dp_ratio, pileup_entropy,
        nm_mean_alt, nm_mean_ref, proper_pair_rate_alt, proper_pair_rate_ref,
        mismatch_rate_alt: true_mismatch_rate_alt,
        cigar_agreement, haplotype_entropy, fragment_support_rate,
        read_end_dist_mean_alt, read_end_dist_mean_ref,
        softclip_rate_ref, softclip_diff,
        isize_mean_alt, isize_mean_ref, isize_dev,
        flank_bq_alt, flank_bq_ref, true_mismatch_rate_alt,
        tandem_repeat_unit: tr_unit, tandem_repeat_length: tr_length,
        mq0_fraction, mq_diff,
    }))
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

fn var(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}

fn entropy_bytes(bytes: &[u8]) -> f64 {
    if bytes.is_empty() { return 0.0; }
    let mut counts: HashMap<u8, usize> = HashMap::new();
    for &b in bytes {
        *counts.entry(b).or_insert(0) += 1;
    }
    let n = bytes.len() as f64;
    counts.values().map(|&c| {
        let p = c as f64 / n;
        -p * p.log2()
    }).sum()
}

fn compute_nbq(quals: &[u8], pairs: &[(usize, i64)], var_pos_0based: i64, window: i64) -> f64 {
    if quals.is_empty() { return 0.0; }
    let lo = var_pos_0based - window;
    let hi = var_pos_0based + window;
    let mut total: u64 = 0;
    let mut count: u64 = 0;
    for &(qp, rp) in pairs {
        if rp < lo { continue; }
        if rp > hi { break; }
        if qp < quals.len() {
            total += quals[qp] as u64;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total as f64 / count as f64 }
}

fn flanking_bq(
    quals: &[u8], seq: &[u8], pairs: &[(usize, i64)],
    var_pos_0based: i64, ref_seq: &[u8], ref_offset: i64, window: i64,
) -> (f64, u64, u64) {
    if quals.is_empty() || seq.is_empty() {
        return (0.0, 0, 0);
    }
    let lo = var_pos_0based - window;
    let hi = var_pos_0based + window;
    let mut bq_sum: u64 = 0;
    let mut bq_count: u64 = 0;
    let mut mismatches: u64 = 0;
    let mut aligned: u64 = 0;
    for &(qp, rp) in pairs {
        if rp < lo { continue; }
        if rp > hi { break; }
        if qp < quals.len() {
            bq_sum += quals[qp] as u64;
            bq_count += 1;
            aligned += 1;
            if !ref_seq.is_empty() {
                let ridx = (rp - ref_offset) as i64;
                if ridx >= 0 && (ridx as usize) < ref_seq.len() {
                    let q_byte = seq[qp].to_ascii_uppercase();
                    let r_byte = ref_seq[ridx as usize].to_ascii_uppercase();
                    if q_byte != r_byte {
                        mismatches += 1;
                    }
                }
            }
        }
    }
    let mean_bq = if bq_count == 0 { 0.0 } else { bq_sum as f64 / bq_count as f64 };
    (mean_bq, mismatches, aligned)
}

fn compute_context(seq: &[u8], pos_in_seq: i64) -> (f64, i64, i64, i64) {
    if seq.is_empty() {
        return (0.0, 1, 1, 1);
    }
    let n = seq.len() as f64;
    let gc_count = seq.iter().filter(|&&b| b == b'G' || b == b'C').count();
    let gc = gc_count as f64 / n;

    let ci = pos_in_seq;
    if ci < 0 || (ci as usize) >= seq.len() {
        return (gc, 1, 1, 1);
    }
    let ci = ci as usize;
    let center = seq[ci];

    let mut homopoly: i64 = 1;
    let mut j = ci as i64 - 1;
    while j >= 0 && seq[j as usize] == center {
        homopoly += 1; j -= 1;
    }
    let mut j = ci + 1;
    while j < seq.len() && seq[j] == center {
        homopoly += 1; j += 1;
    }

    let (tr_unit, tr_length) = find_tandem_repeat(seq, ci, 6);
    (gc, homopoly, tr_unit, tr_length)
}

fn find_tandem_repeat(seq: &[u8], pos: usize, max_unit: usize) -> (i64, i64) {
    let mut best_total: i64 = 1;
    let mut best_unit: i64 = 1;
    for unit_len in 1..=max_unit {
        if pos + unit_len > seq.len() { break; }
        let unit = &seq[pos..pos + unit_len];
        let mut left = pos;
        while left >= unit_len && &seq[left - unit_len..left] == unit {
            left -= unit_len;
        }
        let mut right = pos + unit_len;
        while right + unit_len <= seq.len() && &seq[right..right + unit_len] == unit {
            right += unit_len;
        }
        let total = (right - left) as i64;
        if total > best_total {
            best_total = total;
            best_unit = unit_len as i64;
        }
    }
    (best_unit, best_total)
}
