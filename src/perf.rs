use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;

use crate::benchmark::{run_benchmark, prompts_to_messages, BenchmarkConfig, BenchmarkResult};
use crate::client::{single_user_message, OpenAIClient};
use crate::config::TaskConfig;
use crate::datasets::pad_to_target_chars;

pub struct SweepEntry {
    pub concurrency: usize,
    pub num_requests: usize,
    pub result: BenchmarkResult,
}

pub struct PerfReport {
    pub task_name: String,
    pub base_url: String,
    pub model: Option<String>,
    pub dataset: Option<String>,
    pub max_tokens: u32,
    pub timestamp_secs: u64,
    pub total_duration: f64,
    pub sweep: Vec<SweepEntry>,
}

pub struct PerfSuiteConfig {
    pub concurrencies: Vec<usize>,
    pub n_per_level: usize,
    pub warmup: usize,
    /// Seconds to pause between sweep levels. Lets in-flight requests drain
    /// and gives the server's prefix cache a chance to evict before the next
    /// level inherits warmed lines.
    pub cooldown_secs: f64,
    /// Between sweep levels, fire N unique-content requests to purge the
    /// server's prefix cache via LRU. Set to 0 to disable. Each storm request
    /// is ~`eviction_storm_tokens` tokens, max_tokens=4, so it exercises
    /// prefill only.
    pub eviction_storm_requests: usize,
    pub eviction_storm_tokens: usize,
}

impl Default for PerfSuiteConfig {
    fn default() -> Self {
        Self {
            concurrencies: vec![1, 2, 4, 8],
            n_per_level: 20,
            warmup: 2,
            cooldown_secs: 5.0,
            eviction_storm_requests: 8,
            eviction_storm_tokens: 4000,
        }
    }
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Build a request whose leading tokens are *guaranteed* novel to the server's
/// prefix cache. We seed each request with a unique nonce, then pad with the
/// same nonce repeated so the whole prompt is effectively a fresh radix-tree
/// branch. The model never sees this twice across (or within) a storm.
fn make_storm_message(seed: u64, target_tokens: usize, chars_per_token: f64) -> Value {
    let ratio = if chars_per_token > 0.0 { chars_per_token } else { 4.0 };
    let target_chars = (target_tokens as f64 * ratio) as usize;
    // Two interleaved nonces from a splitmix-like mix so consecutive seeds
    // produce tokenizer-distinct strings (not just adjacent integers).
    let nonce_a = seed;
    let nonce_b = seed
        .wrapping_add(0x9e3779b97f4a7c15)
        .wrapping_mul(0xbf58476d1ce4e5b9);
    let block = format!("evict-{:016x}-{:016x} ", nonce_a, nonce_b);
    let mut s = String::with_capacity(target_chars + block.len());
    while s.len() < target_chars {
        s.push_str(&block);
    }
    s.truncate(target_chars);
    single_user_message(&s)
}

/// Fire `count` unique-content requests in parallel to push the previous
/// level's cache entries out via LRU. Best-effort: errors are swallowed
/// since this is non-measurement traffic.
async fn eviction_storm(
    client: Arc<OpenAIClient>,
    model: Option<&str>,
    count: usize,
    tokens_per_request: usize,
    quiet: bool,
) {
    if count == 0 || tokens_per_request == 0 {
        return;
    }
    if !quiet {
        println!(
            "    (eviction storm: {} unique prompts × ~{} tokens)",
            count, tokens_per_request
        );
    }

    let started = Instant::now();
    // Base seed from wall-clock + process id (well, just nanos for now) so
    // storms across multiple suite runs don't accidentally hit the same
    // nonces and remain cached.
    let base_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    let chars_per_token = client.chars_per_token;
    let mut tasks = Vec::with_capacity(count);
    for i in 0..count {
        let client = Arc::clone(&client);
        let model = model.map(String::from);
        let seed = base_seed.wrapping_add(i as u64);
        tasks.push(tokio::spawn(async move {
            let msg = make_storm_message(seed, tokens_per_request, chars_per_token);
            // max_tokens=4 keeps decode trivial; prefill is what loads the
            // new cache entries we care about.
            let _ = client.complete(&msg, model.as_deref(), 4).await;
        }));
    }
    for t in tasks {
        let _ = t.await;
    }

    if !quiet {
        println!(
            "    eviction storm complete in {:.1}s",
            started.elapsed().as_secs_f64()
        );
    }
}

pub async fn run_perf_suite(
    client: Arc<OpenAIClient>,
    messages_list: Vec<Value>,
    task: &TaskConfig,
    suite: &PerfSuiteConfig,
    quiet: bool,
) -> Result<PerfReport> {
    let suite_start = Instant::now();
    let timestamp_secs = now_secs();

    if messages_list.is_empty() {
        anyhow::bail!("perf suite requires at least one prompt");
    }

    let mut sweep = Vec::new();
    for (idx, &c) in suite.concurrencies.iter().enumerate() {
        // Between-level cleanup (not before the first one): first push the
        // previous level's cache entries out via LRU, then a short pause to
        // let any tail requests drain.
        if idx > 0 {
            eviction_storm(
                Arc::clone(&client),
                task.model.as_deref(),
                suite.eviction_storm_requests,
                suite.eviction_storm_tokens,
                quiet,
            )
            .await;
            if suite.cooldown_secs > 0.0 {
                if !quiet {
                    println!("    (cooling down {:.0}s before next level)", suite.cooldown_secs);
                }
                tokio::time::sleep(std::time::Duration::from_secs_f64(suite.cooldown_secs)).await;
            }
        }

        if !quiet {
            println!(
                "[{}/{}] sweep c={} (n={}, warmup={})",
                idx + 1,
                suite.concurrencies.len(),
                c,
                suite.n_per_level,
                suite.warmup,
            );
        }

        let messages_slice: Vec<Value> = messages_list
            .iter()
            .cycle()
            .take(suite.n_per_level)
            .cloned()
            .collect();

        let pb = if !quiet {
            let bar = ProgressBar::new(suite.n_per_level as u64);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template("  [{pos}/{len}] {msg}")
                    .unwrap(),
            );
            Some(bar)
        } else {
            None
        };

        let bench_cfg = BenchmarkConfig {
            model: task.model.clone(),
            max_tokens: task.max_tokens,
            concurrency: c,
            warmup: suite.warmup,
            max_retries: task.retries,
            request_rate: task.request_rate,
        };

        let result = run_benchmark(Arc::clone(&client), messages_slice, bench_cfg, pb).await;

        if !quiet {
            println!(
                "    {:.1}s  TPS={:.1}  TTFT p50={:.3}s p99={:.3}s  errors={}",
                result.total_duration,
                result.output_tps,
                result.ttft_p50,
                result.ttft_p99,
                result.num_errors,
            );
        }

        sweep.push(SweepEntry {
            concurrency: c,
            num_requests: suite.n_per_level,
            result,
        });
    }

    let total_duration = suite_start.elapsed().as_secs_f64();

    Ok(PerfReport {
        task_name: task.name.clone(),
        base_url: task.base_url.clone(),
        model: task.model.clone(),
        dataset: task.dataset.clone(),
        max_tokens: task.max_tokens,
        timestamp_secs,
        total_duration,
        sweep,
    })
}

enum KneeResult {
    Optimal(usize),
    SweepHigher,
    Insufficient,
}

fn detect_knee(sweep: &[SweepEntry]) -> KneeResult {
    if sweep.len() < 2 {
        return KneeResult::Insufficient;
    }
    let threshold = 1.2;
    for w in sweep.windows(2) {
        let prev_tps = w[0].result.output_tps;
        if prev_tps <= 0.0 {
            return KneeResult::Insufficient;
        }
        let gain = w[1].result.output_tps / prev_tps;
        if gain < threshold {
            return KneeResult::Optimal(w[0].concurrency);
        }
    }
    KneeResult::SweepHigher
}

fn format_duration(secs: f64) -> String {
    let total_s = secs.round() as u64;
    let m = total_s / 60;
    let s = total_s % 60;
    if m > 0 {
        format!("{}m {}s", m, s)
    } else {
        format!("{}s", s)
    }
}

fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn epoch_to_ymd_hms(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let time_of_day = secs % 86_400;
    let h = (time_of_day / 3600) as u32;
    let m = ((time_of_day % 3600) / 60) as u32;
    let s = (time_of_day % 60) as u32;

    let mut days = (secs / 86_400) as i64;
    let mut year = 1970u32;
    loop {
        let year_days: i64 = if is_leap(year) { 366 } else { 365 };
        if days < year_days {
            break;
        }
        days -= year_days;
        year += 1;
    }
    let months_in_year: [i64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1u32;
    for &md in months_in_year.iter() {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, (days + 1) as u32, h, m, s)
}

pub fn format_filename_timestamp(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = epoch_to_ymd_hms(secs);
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, mo, d, h, mi, s)
}

pub fn format_human_timestamp(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = epoch_to_ymd_hms(secs);
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC", y, mo, d, h, mi, s)
}

pub fn format_markdown(report: &PerfReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    let _ = writeln!(out, "# yabench performance report");
    let _ = writeln!(out);
    let _ = writeln!(out, "| | |");
    let _ = writeln!(out, "|---|---|");
    let _ = writeln!(out, "| **Task** | `{}` |", report.task_name);
    let _ = writeln!(out, "| **Endpoint** | `{}` |", report.base_url);
    let _ = writeln!(
        out,
        "| **Model** | `{}` |",
        report.model.as_deref().unwrap_or("(default)")
    );
    let _ = writeln!(
        out,
        "| **Dataset** | `{}` |",
        report.dataset.as_deref().unwrap_or("(synthetic)")
    );
    let _ = writeln!(out, "| **Max output tokens** | {} |", report.max_tokens);
    let _ = writeln!(
        out,
        "| **Generated** | {} |",
        format_human_timestamp(report.timestamp_secs)
    );
    let _ = writeln!(
        out,
        "| **Total suite duration** | {} |",
        format_duration(report.total_duration)
    );
    let _ = writeln!(out);

    let n = report
        .sweep
        .first()
        .map(|e| e.num_requests)
        .unwrap_or(0);

    let _ = writeln!(out, "## Concurrency sweep");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Each level runs n={} requests with warmup. Same prompts across levels for comparability.",
        n,
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| c | wall (s) | Output TPS | Req/s | TTFT p50 | TTFT p95 | TTFT p99 | E2E p50 | E2E p95 | E2E p99 | ITL (ms) | Errors |"
    );
    let _ = writeln!(
        out,
        "|--:|---------:|-----------:|------:|---------:|---------:|---------:|--------:|--------:|--------:|---------:|-------:|"
    );
    for entry in &report.sweep {
        let r = &entry.result;
        let _ = writeln!(
            out,
            "| {} | {:.1} | {:.1} | {:.2} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} | {:.2} | {:.0} | {} |",
            entry.concurrency,
            r.total_duration,
            r.output_tps,
            r.requests_per_second,
            r.ttft_p50,
            r.ttft_p95,
            r.ttft_p99,
            r.e2e_p50,
            r.e2e_p95,
            r.e2e_p99,
            r.inter_token_latency_mean * 1000.0,
            r.num_errors,
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Scaling analysis");
    let _ = writeln!(out);
    if report.sweep.len() >= 2 {
        for w in report.sweep.windows(2) {
            let prev = &w[0];
            let curr = &w[1];
            let tps_gain = if prev.result.output_tps > 0.0 {
                curr.result.output_tps / prev.result.output_tps
            } else {
                0.0
            };
            let ttft_ratio = if prev.result.ttft_p50 > 0.0 {
                curr.result.ttft_p50 / prev.result.ttft_p50
            } else {
                0.0
            };
            let _ = writeln!(
                out,
                "- **c={} → c={}**: throughput ×{:.2}, TTFT p50 ×{:.2}",
                prev.concurrency, curr.concurrency, tps_gain, ttft_ratio,
            );
        }
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "## Recommendation");
    let _ = writeln!(out);
    match detect_knee(&report.sweep) {
        KneeResult::Optimal(c) => {
            let _ = writeln!(
                out,
                "**Optimal concurrency: c={}.** Above this point, throughput gains drop below 20% per doubling — additional concurrency mostly buys latency, not work.",
                c,
            );
        }
        KneeResult::SweepHigher => {
            let _ = writeln!(
                out,
                "**The sweep did not reach saturation.** Throughput was still scaling ≥20% per doubling at the top of the range (c={}). Rerun a wider sweep manually (e.g. `yabench {} -n {} -c 16` and `-c 32`) to find the knee.",
                report.sweep.last().unwrap().concurrency,
                report.task_name,
                n,
            );
        }
        KneeResult::Insufficient => {
            let _ = writeln!(out, "Not enough sweep data to recommend a concurrency level.");
        }
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Caveat: percentile reliability");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Each level has n={} samples. p50 and p75 are trustworthy; p95 is the 1-2 slowest requests; p99 is essentially the single slowest and may be a fluke. For honest tail latency, rerun the chosen concurrency level with at least n=200.",
        n,
    );

    out
}

// ---------------------------------------------------------------------------
// Performance matrix: input_tokens × output_tokens × concurrency
// ---------------------------------------------------------------------------

pub struct MatrixCell {
    pub input_tokens: usize,
    pub output_tokens: u32,
    pub concurrency: usize,
    pub result: BenchmarkResult,
}

pub struct MatrixReport {
    pub task_name: String,
    pub base_url: String,
    pub model: Option<String>,
    pub n_per_cell: usize,
    pub timestamp_secs: u64,
    pub total_duration: f64,
    pub cells: Vec<MatrixCell>,
}

pub struct MatrixConfig {
    pub input_tokens: Vec<usize>,
    pub output_tokens: Vec<u32>,
    pub concurrencies: Vec<usize>,
    pub n_per_cell: usize,
}

impl Default for MatrixConfig {
    fn default() -> Self {
        Self {
            input_tokens: vec![1_000, 4_000, 16_000, 64_000, 128_000],
            output_tokens: vec![256, 1_000, 4_000, 16_000, 64_000],
            concurrencies: vec![1, 4, 8],
            n_per_cell: 10,
        }
    }
}

fn format_token_count(n: usize) -> String {
    if n >= 1_000_000 && n % 1_000_000 == 0 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 && n % 1_000 == 0 {
        format!("{}K", n / 1_000)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

pub async fn run_perf_matrix(
    client: Arc<OpenAIClient>,
    corpus: &[String],
    task: &TaskConfig,
    cfg: &MatrixConfig,
    quiet: bool,
) -> Result<MatrixReport> {
    let suite_start = Instant::now();
    let timestamp_secs = now_secs();

    let total_cells = cfg.input_tokens.len() * cfg.output_tokens.len() * cfg.concurrencies.len();
    let mut cells = Vec::with_capacity(total_cells);
    let mut cell_idx = 0usize;

    for &input_tok in &cfg.input_tokens {
        let target_chars = (input_tok as f64 * client.chars_per_token) as usize;
        let prompts = pad_to_target_chars(corpus, cfg.n_per_cell, target_chars);
        let messages_list = prompts_to_messages(prompts);

        for &output_tok in &cfg.output_tokens {
            for &c in &cfg.concurrencies {
                cell_idx += 1;
                if !quiet {
                    println!(
                        "[{}/{}] input={} output={} c={}",
                        cell_idx,
                        total_cells,
                        format_token_count(input_tok),
                        format_token_count(output_tok as usize),
                        c,
                    );
                }

                let msgs: Vec<Value> = messages_list
                    .iter()
                    .cycle()
                    .take(cfg.n_per_cell)
                    .cloned()
                    .collect();

                let bench_cfg = BenchmarkConfig {
                    model: task.model.clone(),
                    max_tokens: output_tok,
                    concurrency: c,
                    warmup: 0,
                    max_retries: task.retries,
                    request_rate: task.request_rate,
                };

                let result = run_benchmark(Arc::clone(&client), msgs, bench_cfg, None).await;

                if !quiet {
                    println!(
                        "    {:.1}s  TPS={:.1}  TTFT p50={:.3}s  errors={}",
                        result.total_duration,
                        result.output_tps,
                        result.ttft_p50,
                        result.num_errors,
                    );
                }

                cells.push(MatrixCell {
                    input_tokens: input_tok,
                    output_tokens: output_tok,
                    concurrency: c,
                    result,
                });
            }
        }
    }

    let total_duration = suite_start.elapsed().as_secs_f64();

    Ok(MatrixReport {
        task_name: task.name.clone(),
        base_url: task.base_url.clone(),
        model: task.model.clone(),
        n_per_cell: cfg.n_per_cell,
        timestamp_secs,
        total_duration,
        cells,
    })
}

pub fn format_matrix_markdown(report: &MatrixReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    let _ = writeln!(out, "# yabench performance matrix");
    let _ = writeln!(out);
    let _ = writeln!(out, "| | |");
    let _ = writeln!(out, "|---|---|");
    let _ = writeln!(out, "| **Task** | `{}` |", report.task_name);
    let _ = writeln!(out, "| **Endpoint** | `{}` |", report.base_url);
    let _ = writeln!(
        out,
        "| **Model** | `{}` |",
        report.model.as_deref().unwrap_or("(default)")
    );
    let _ = writeln!(out, "| **Requests per cell** | {} |", report.n_per_cell);
    let _ = writeln!(
        out,
        "| **Generated** | {} |",
        format_human_timestamp(report.timestamp_secs)
    );
    let _ = writeln!(
        out,
        "| **Total duration** | {} |",
        format_duration(report.total_duration)
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Results");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| Input | Output | c | Output TPS | Req/s | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | Prefill TPS | Errors |"
    );
    let _ = writeln!(
        out,
        "|------:|-------:|--:|-----------:|------:|---------:|---------:|--------:|--------:|------------:|-------:|"
    );

    for cell in &report.cells {
        let r = &cell.result;
        let _ = writeln!(
            out,
            "| {} | {} | {} | {:.1} | {:.2} | {:.3} | {:.3} | {:.2} | {:.2} | {:.0} | {} |",
            format_token_count(cell.input_tokens),
            format_token_count(cell.output_tokens as usize),
            cell.concurrency,
            r.output_tps,
            r.requests_per_second,
            r.ttft_p50,
            r.ttft_p99,
            r.e2e_p50,
            r.e2e_p99,
            r.prefill_tps_mean,
            r.num_errors,
        );
    }

    out
}
