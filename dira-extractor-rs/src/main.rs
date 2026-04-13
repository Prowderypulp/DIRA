//! DIRA feature extractor — Rust port (v2).
//!
//! Key fix vs v1: BAM/VCF/FASTA handles are opened **once per worker thread**
//! and reused across many regions, instead of being reopened per region.
//! This eliminates the per-region index-load + file-open overhead that
//! dominated v1 runtime.
//!
//! Also: per-thread CSV buffering, then merged at end. No mutex contention
//! on the writer.

mod cigar;
mod features;
mod region;

use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use rust_htslib::bam::IndexedReader as BamReader;
use rust_htslib::bcf::IndexedReader as BcfReader;
use rust_htslib::faidx::Reader as FaReader;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::features::{FeatureRow, HEADER};
use crate::region::{process_region, Region};

#[derive(Parser, Debug)]
#[command(version, about = "DIRA feature extractor (Rust v2)")]
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

fn worker_thread(
    thread_id: usize,
    rx: Receiver<Region>,
    bam_path: String,
    vcf_path: String,
    ref_path: String,
    tmp_path: PathBuf,
    done_counter: Arc<AtomicUsize>,
    total: usize,
) -> Result<()> {
    // Open handles ONCE for this thread's lifetime.
    let mut bam = BamReader::from_path(&bam_path)
        .with_context(|| format!("[t{}] opening BAM", thread_id))?;
    let mut vcf = BcfReader::from_path(&vcf_path)
        .with_context(|| format!("[t{}] opening VCF", thread_id))?;
    let fa = FaReader::from_path(&ref_path)
        .with_context(|| format!("[t{}] opening FASTA", thread_id))?;

    let file = File::create(&tmp_path)
        .with_context(|| format!("[t{}] creating tmp {}", thread_id, tmp_path.display()))?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(BufWriter::with_capacity(1 << 20, file));

    while let Ok(region) = rx.recv() {
        match process_region(&region, &mut bam, &mut vcf, &fa) {
            Ok(rows) => {
                for row in rows {
                    writer.write_record(&row.to_record())?;
                }
            }
            Err(e) => {
                eprintln!("[t{}] error in {}:{}-{}: {:#}",
                    thread_id, region.chrom, region.start, region.end, e);
            }
        }
        let n = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 100 == 0 || n == total {
            eprintln!("  [{}/{}] regions done", n, total);
        }
    }

    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    // Sanity-check FASTA index.
    let _ = FaReader::from_path(&args.ref_)
        .with_context(|| format!("opening FASTA {}", args.ref_))?;

    let regions = build_regions(&args.ref_, &args.chroms, args.chunk_size)?;
    let n_workers = if args.workers == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        args.workers
    };

    eprintln!(
        "[dira-rs] {} chromosomes -> {} regions, {} workers",
        args.chroms.len(),
        regions.len(),
        n_workers,
    );

    // Set up channel + worker threads.
    let (tx, rx): (Sender<Region>, Receiver<Region>) = bounded(n_workers * 4);
    let done_counter = Arc::new(AtomicUsize::new(0));
    let total = regions.len();

    let tmp_dir = tempfile::tempdir()?;
    let tmp_paths: Vec<PathBuf> = (0..n_workers)
        .map(|i| tmp_dir.path().join(format!("worker_{}.csv", i)))
        .collect();

    let mut handles = Vec::with_capacity(n_workers);
    for i in 0..n_workers {
        let rx = rx.clone();
        let bam_path = args.bam.clone();
        let vcf_path = args.vcf.clone();
        let ref_path = args.ref_.clone();
        let tmp_path = tmp_paths[i].clone();
        let done_counter = Arc::clone(&done_counter);
        let h = thread::Builder::new()
            .name(format!("worker-{}", i))
            .spawn(move || {
                worker_thread(i, rx, bam_path, vcf_path, ref_path, tmp_path, done_counter, total)
            })?;
        handles.push(h);
    }
    drop(rx); // workers hold their own clones

    // Dispatch regions.
    for region in regions {
        tx.send(region).unwrap();
    }
    drop(tx); // signal workers no more work

    // Wait for workers.
    for h in handles {
        h.join().expect("worker panicked")?;
    }

    // Merge per-thread tmp CSVs into the final output.
    eprintln!("[dira-rs] merging {} thread outputs...", n_workers);
    let out_file = File::create(&args.out)
        .with_context(|| format!("creating {}", args.out))?;
    let mut out = BufWriter::with_capacity(1 << 20, out_file);

    // Header
    {
        let mut header_csv = csv::WriterBuilder::new().from_writer(&mut out);
        header_csv.write_record(HEADER)?;
        header_csv.flush()?;
    }

    for tmp_path in &tmp_paths {
        if tmp_path.exists() {
            let mut f = File::open(tmp_path)?;
            std::io::copy(&mut f, &mut out)?;
        }
    }
    out.flush()?;

    eprintln!("[dira-rs] done -> {} ({:.1}s)", args.out, t0.elapsed().as_secs_f64());
    Ok(())
}
