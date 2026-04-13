//! DIRA feature extractor — Rust port.
//!
//! CLI mirrors the v6 Python extractor:
//!   --vcf <path>  --bam <path>  --ref <path>  --out <path>
//!   [--chroms chr1 chr2 ...]  [--workers N]  [--chunk-size BP]

mod cigar;
mod features;
mod region;

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use rust_htslib::faidx::Reader as FaReader;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;
use std::time::Instant;

use crate::features::{FeatureRow, HEADER};
use crate::region::{process_region, Region};

#[derive(Parser, Debug)]
#[command(version, about = "DIRA feature extractor (Rust)")]
struct Args {
    #[arg(long)]
    vcf: String,
    #[arg(long)]
    bam: String,
    #[arg(long, name = "ref")]
    ref_: String,
    #[arg(long)]
    out: String,
    #[arg(long, num_args = 1.., default_values_t = (1..=22).map(|i| format!("chr{}", i)).collect::<Vec<_>>())]
    chroms: Vec<String>,
    #[arg(long, default_value_t = 0)]
    workers: usize,
    #[arg(long = "chunk-size", default_value_t = 1_000_000u64)]
    chunk_size: u64,
}

fn build_regions(ref_path: &str, chroms: &[String], chunk_size: u64) -> Result<Vec<Region>> {
    // rust-htslib's faidx Reader doesn't expose lengths directly; read .fai.
    let fai_path = format!("{}.fai", ref_path);
    let fai = std::fs::read_to_string(&fai_path)
        .with_context(|| format!("reading {}", fai_path))?;
    let mut lengths = std::collections::HashMap::new();
    for line in fai.lines() {
        let mut it = line.split('\t');
        let name = it.next().unwrap_or("").to_string();
        let len: u64 = it.next().unwrap_or("0").parse().unwrap_or(0);
        lengths.insert(name, len);
    }

    let mut regions = Vec::new();
    for chrom in chroms {
        let len = match lengths.get(chrom) {
            Some(&l) => l,
            None => {
                eprintln!("[warn] {} not in reference, skipping", chrom);
                continue;
            }
        };
        let mut s: u64 = 1;
        while s <= len {
            let e = std::cmp::min(s + chunk_size - 1, len);
            regions.push(Region {
                chrom: chrom.clone(),
                start: s,
                end: e,
            });
            s += chunk_size;
        }
    }
    Ok(regions)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    // Sanity-check the FASTA index exists.
    let _ = FaReader::from_path(&args.ref_)
        .with_context(|| format!("opening FASTA {}", args.ref_))?;

    let regions = build_regions(&args.ref_, &args.chroms, args.chunk_size)?;
    eprintln!(
        "[dira-rs] {} chromosomes -> {} regions, {} workers",
        args.chroms.len(),
        regions.len(),
        if args.workers == 0 { num_cpus() } else { args.workers }
    );

    if args.workers > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.workers)
            .build_global()
            .ok();
    }

    let out_file = File::create(&args.out)
        .with_context(|| format!("creating {}", args.out))?;
    let writer = Mutex::new(csv::Writer::from_writer(BufWriter::with_capacity(1 << 20, out_file)));

    // Header
    {
        let mut w = writer.lock().unwrap();
        w.write_record(HEADER)?;
    }

    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = regions.len();

    regions.par_iter().try_for_each(|reg| -> Result<()> {
        let rows: Vec<FeatureRow> = process_region(reg, &args.bam, &args.vcf, &args.ref_)?;
        if !rows.is_empty() {
            let mut w = writer.lock().unwrap();
            for row in rows {
                w.write_record(&row.to_record())?;
            }
        }
        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n % 50 == 0 || n == total {
            eprintln!("  [{}/{}] regions done", n, total);
        }
        Ok(())
    })?;

    {
        let mut w = writer.lock().unwrap();
        w.flush()?;
    }

    eprintln!("[dira-rs] done -> {} ({:.1}s)", args.out, t0.elapsed().as_secs_f64());
    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}
