use std::sync::Arc;
use std::time::Instant;
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::client::{single_user_message, OpenAIClient, RequestMetrics};

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub num_requests: usize,
    pub num_completed: usize,
    pub num_errors: usize,
    pub total_duration: f64,

    // TTFT stats (seconds)
    pub ttft_mean: f64,
    pub ttft_p50: f64,
    pub ttft_p75: f64,
    pub ttft_p90: f64,
    pub ttft_p95: f64,
    pub ttft_p98: f64,
    pub ttft_p99: f64,
    pub ttft_max: f64,

    // E2E latency stats
    pub e2e_mean: f64,
    pub e2e_p50: f64,
    pub e2e_p95: f64,
    pub e2e_p99: f64,

    pub inter_token_latency_mean: f64,
    /// Time per output token, excluding the first token: (E2E − TTFT)/(out−1),
    /// averaged over requests. Matches vLLM bench / ais_bench "TPOT".
    pub tpot_mean: f64,

    // Throughput
    pub output_tps: f64,
    pub requests_per_second: f64,
    pub prefill_tps_mean: f64,

    // Token counts
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub input_tokens_estimated_count: usize,
    pub output_tokens_estimated_count: usize,

    pub errors: Vec<String>,
    pub request_metrics: Vec<RequestMetrics>,
}

impl BenchmarkResult {
    fn new(num_requests: usize) -> Self {
        Self {
            num_requests,
            num_completed: 0,
            num_errors: 0,
            total_duration: 0.0,
            ttft_mean: 0.0,
            ttft_p50: 0.0,
            ttft_p75: 0.0,
            ttft_p90: 0.0,
            ttft_p95: 0.0,
            ttft_p98: 0.0,
            ttft_p99: 0.0,
            ttft_max: 0.0,
            e2e_mean: 0.0,
            e2e_p50: 0.0,
            e2e_p95: 0.0,
            e2e_p99: 0.0,
            inter_token_latency_mean: 0.0,
            tpot_mean: 0.0,
            output_tps: 0.0,
            requests_per_second: 0.0,
            prefill_tps_mean: 0.0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            input_tokens_estimated_count: 0,
            output_tokens_estimated_count: 0,
            errors: vec![],
            request_metrics: vec![],
        }
    }
}

fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = (sorted.len() - 1) as f64 * p / 100.0;
    let f = k.floor() as usize;
    let c = (f + 1).min(sorted.len() - 1);
    sorted[f] + (sorted[c] - sorted[f]) * (k - f as f64)
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn compute_stats(result: &mut BenchmarkResult) {
    let completed: Vec<&RequestMetrics> = result
        .request_metrics
        .iter()
        .filter(|m| m.error.is_none())
        .collect();

    if completed.is_empty() {
        return;
    }

    let ttfts: Vec<f64> = completed.iter().filter_map(|m| m.ttft).collect();
    if !ttfts.is_empty() {
        result.ttft_mean = mean(&ttfts);
        result.ttft_p50 = percentile(&ttfts, 50.0);
        result.ttft_p75 = percentile(&ttfts, 75.0);
        result.ttft_p90 = percentile(&ttfts, 90.0);
        result.ttft_p95 = percentile(&ttfts, 95.0);
        result.ttft_p98 = percentile(&ttfts, 98.0);
        result.ttft_p99 = percentile(&ttfts, 99.0);
        result.ttft_max = ttfts.iter().copied().fold(0.0_f64, f64::max);
    }

    let e2e: Vec<f64> = completed.iter().map(|m| m.total_time).collect();
    result.e2e_mean = mean(&e2e);
    result.e2e_p50 = percentile(&e2e, 50.0);
    result.e2e_p95 = percentile(&e2e, 95.0);
    result.e2e_p99 = percentile(&e2e, 99.0);

    let all_itl: Vec<f64> = completed
        .iter()
        .flat_map(|m| m.inter_token_latencies.iter().copied())
        .collect();
    result.inter_token_latency_mean = mean(&all_itl);

    // TPOT: decode time per output token excluding the first, per request.
    // Needs TTFT and at least 2 output tokens to be meaningful.
    let tpots: Vec<f64> = completed
        .iter()
        .filter_map(|m| {
            let ttft = m.ttft?;
            if m.output_tokens >= 2 && m.total_time > ttft {
                Some((m.total_time - ttft) / (m.output_tokens - 1) as f64)
            } else {
                None
            }
        })
        .collect();
    result.tpot_mean = mean(&tpots);

    result.total_input_tokens = completed.iter().map(|m| m.input_tokens).sum();
    result.total_output_tokens = completed.iter().map(|m| m.output_tokens).sum();
    result.input_tokens_estimated_count = completed.iter().filter(|m| m.input_tokens_estimated).count();
    result.output_tokens_estimated_count = completed.iter().filter(|m| m.output_tokens_estimated).count();

    if result.total_duration > 0.0 {
        result.output_tps = result.total_output_tokens as f64 / result.total_duration;
        result.requests_per_second = result.num_completed as f64 / result.total_duration;
    }

    let prefill_vals: Vec<f64> = completed
        .iter()
        .map(|m| m.prefill_tps())
        .filter(|&v| v > 0.0)
        .collect();
    result.prefill_tps_mean = mean(&prefill_vals);
}

pub struct BenchmarkConfig {
    pub model: Option<String>,
    pub max_tokens: u32,
    pub concurrency: usize,
    pub warmup: usize,
    pub max_retries: u32,
    /// Steady-state arrival rate (req/s). None = burst up to `concurrency`.
    /// Some(r) = launch one request every 1/r seconds (concurrency still caps
    /// in-flight requests). Matches vLLM bench / ais_bench arrival model.
    pub request_rate: Option<f64>,
}

pub async fn run_benchmark(
    client: Arc<OpenAIClient>,
    messages_list: Vec<Value>,
    config: BenchmarkConfig,
    progress: Option<indicatif::ProgressBar>,
) -> BenchmarkResult {
    let num_prompts = messages_list.len();
    let semaphore = Arc::new(Semaphore::new(config.concurrency));
    let model = config.model.map(Arc::new);
    let max_tokens = config.max_tokens;
    let max_retries = config.max_retries;
    // Fixed inter-arrival interval when a request rate is set. Filtered to a
    // positive, finite rate; otherwise None = burst arrival.
    let arrival_interval = config
        .request_rate
        .filter(|r| r.is_finite() && *r > 0.0)
        .map(|r| std::time::Duration::from_secs_f64(1.0 / r));

    // Warm-up phase
    if config.warmup > 0 {
        let warmup_msgs: Vec<_> = messages_list
            .iter()
            .cycle()
            .take(config.warmup)
            .cloned()
            .collect();
        let warmup_sem = Arc::new(Semaphore::new(config.concurrency));
        let mut warmup_tasks = vec![];

        for msgs in warmup_msgs {
            let client = Arc::clone(&client);
            let sem = Arc::clone(&warmup_sem);
            let model = model.clone();

            warmup_tasks.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                client
                    .complete(&msgs, model.as_ref().map(|s| s.as_str()), max_tokens)
                    .await
            }));
        }

        for task in warmup_tasks {
            let _ = task.await;
        }
    }

    // Benchmark phase
    let start = Instant::now();
    let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let error_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let mut tasks = vec![];

    for (i, msgs) in messages_list.iter().cloned().enumerate() {
        // Pace arrivals at a fixed interval when a rate is configured. The
        // semaphore still caps concurrent in-flight requests; pacing only
        // controls *when* each request is launched, turning the synchronized
        // burst into steady-state arrival (matching vLLM bench / ais_bench).
        if i > 0 {
            if let Some(interval) = arrival_interval {
                tokio::time::sleep(interval).await;
            }
        }

        let client = Arc::clone(&client);
        let sem = Arc::clone(&semaphore);
        let model = model.clone();
        let pb = progress.clone();
        let completed = Arc::clone(&completed_count);
        let errors = Arc::clone(&error_count);

        tasks.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();

            let mut metrics = client
                .complete(&msgs, model.as_ref().map(|s| s.as_str()), max_tokens)
                .await;

            // Retry logic with exponential backoff
            if metrics.error.is_some() && max_retries > 0 {
                for attempt in 0..max_retries {
                    let wait_ms = 100u64 * (1 << attempt);
                    tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;

                    metrics = client
                        .complete(&msgs, model.as_ref().map(|s| s.as_str()), max_tokens)
                        .await;

                    if metrics.error.is_none() {
                        break;
                    }
                }
            }

            let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let is_err = metrics.error.is_some();
            if is_err {
                errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }

            if let Some(pb) = pb {
                let err_count = errors.load(std::sync::atomic::Ordering::Relaxed);
                let tps = if metrics.total_time > 0.0 {
                    metrics.output_tokens as f64 / metrics.total_time
                } else {
                    0.0
                };
                let ttft_str = metrics
                    .ttft
                    .map(|t| format!("{:.3}s", t))
                    .unwrap_or_else(|| "N/A".to_string());
                pb.set_message(format!(
                    "TPS: {:.1} | TTFT: {} | Errors: {}",
                    tps, ttft_str, err_count
                ));
                pb.set_position(done as u64);
            }

            metrics
        }));
    }

    let mut all_metrics = vec![];
    for task in tasks {
        if let Ok(m) = task.await {
            all_metrics.push(m);
        }
    }

    let total_duration = start.elapsed().as_secs_f64();
    let mut result = BenchmarkResult::new(num_prompts);
    result.total_duration = total_duration;

    for m in &all_metrics {
        if m.error.is_some() {
            result.num_errors += 1;
            result.errors.push(m.error.clone().unwrap());
        } else {
            result.num_completed += 1;
        }
    }

    result.request_metrics = all_metrics;
    compute_stats(&mut result);

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    result
}

/// Wrap each plain-text prompt as a single-user-message conversation.
pub fn prompts_to_messages(prompts: Vec<String>) -> Vec<Value> {
    prompts.into_iter().map(|p| single_user_message(&p)).collect()
}
