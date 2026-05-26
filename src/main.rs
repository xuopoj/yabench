mod benchmark;
mod client;
mod config;
mod datasets;
mod iam;
mod perf;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use console::Style;
use indicatif::{ProgressBar, ProgressStyle};

use benchmark::{run_benchmark, BenchmarkConfig, BenchmarkResult, prompts_to_messages};
use client::OpenAIClient;
use config::{find_config, load_config, parse_size, parse_size_u32, AuthProvider, TaskConfig};
use datasets::{
    download_all_datasets, download_builtin, load_corpus_for_padding, load_dataset,
    load_multi_turn, pad_to_target_chars,
};

/// Default corpus used when no dataset is configured — small ShareGPT-GPT4
/// subset, real conversations, downloads on first use.
const DEFAULT_CORPUS: &str = "sharegpt-small";

#[derive(Parser, Debug)]
#[command(
    name = "yabench",
    about = "Benchmark OpenAI-compatible LLM APIs",
    long_about = None,
)]
struct Cli {
    /// Task name from config file
    task: Option<String>,

    /// Config file path (default: yabench.yaml)
    #[arg(long)]
    config: Option<PathBuf>,

    /// List available tasks from config
    #[arg(long)]
    list: bool,

    /// Download dataset(s). Use without value for all, or specify a name.
    #[arg(long, value_name = "DATASET", num_args = 0..=1, default_missing_value = "all")]
    download: Option<String>,

    /// Base URL for the API
    #[arg(long)]
    base_url: Option<String>,

    /// Model name
    #[arg(long)]
    model: Option<String>,

    /// Auth token (X-Auth-Token header). Env: YABENCH_TOKEN
    #[arg(long, env = "YABENCH_TOKEN")]
    token: Option<String>,

    /// API key (Bearer token). Env: OPENAI_API_KEY
    #[arg(long, env = "OPENAI_API_KEY")]
    api_key: Option<String>,

    /// Number of requests
    #[arg(short = 'n', long)]
    num_requests: Option<usize>,

    /// Concurrent requests
    #[arg(short = 'c', long)]
    concurrency: Option<usize>,

    /// Max tokens per response (accepts 4K, 32K, 128K notation)
    #[arg(long, value_parser = parse_size_u32)]
    max_tokens: Option<u32>,

    /// Approximate input tokens per prompt (accepts 4K, 32K, 128K, 1M notation)
    #[arg(long, value_parser = parse_size)]
    input_tokens: Option<usize>,

    /// Dataset name or file path
    #[arg(long)]
    dataset: Option<String>,

    /// Don't shuffle dataset
    #[arg(long)]
    no_shuffle: bool,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Request timeout in seconds
    #[arg(long)]
    timeout: Option<f64>,

    /// Output file (.json or .csv)
    #[arg(short = 'o', long)]
    output: Option<String>,

    /// Suppress progress output
    #[arg(short = 'q', long)]
    quiet: bool,

    /// Disable SSL verification
    #[arg(short = 'k', long)]
    no_verify_ssl: bool,

    /// Print request/response details for debugging
    #[arg(long)]
    debug: bool,

    /// Print prompts from a dataset and exit (e.g. --print-dataset sample-en)
    #[arg(long, value_name = "DATASET")]
    print_dataset: Option<String>,

    /// List available datasets
    #[arg(long)]
    list_datasets: bool,

    /// Create a sample yabench.yaml in the current directory
    #[arg(long)]
    init: bool,

    /// Print the full usage guide in markdown
    #[arg(long)]
    guide: bool,

    /// Number of warmup requests
    #[arg(long, default_value = "0")]
    warmup: usize,

    /// Max retries on error
    #[arg(long, default_value = "0")]
    retries: u32,

    /// Send a single chat message and stream the response (e.g. --chat "hello")
    #[arg(long, value_name = "PROMPT")]
    chat: Option<String>,

    /// Run the perf suite (concurrency sweep c=1,2,4,8) and write a markdown report.
    /// Optional path; defaults to perf-report-{timestamp}.md
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    perf_report: Option<String>,

    /// Run a performance matrix: sweep input_tokens × output_tokens × concurrency.
    /// Produces a flat markdown table. Optional path; defaults to perf-matrix-{timestamp}.md
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    perf_matrix: Option<String>,

    /// Requests per cell in the perf matrix (default: 10)
    #[arg(long, default_value = "10")]
    matrix_n: usize,

    /// Input token sizes for the perf matrix (comma-separated, e.g. "1K,4K,32K,128K")
    #[arg(long, value_name = "SIZES", value_delimiter = ',', value_parser = parse_size)]
    matrix_input: Option<Vec<usize>>,

    /// Output token sizes for the perf matrix (comma-separated, e.g. "256,1K,8K,64K")
    #[arg(long, value_name = "SIZES", value_delimiter = ',', value_parser = parse_size_u32)]
    matrix_output: Option<Vec<u32>>,

    /// Concurrency levels for the perf matrix (comma-separated, e.g. "1,2,4,8,16")
    #[arg(long, value_name = "LEVELS", value_delimiter = ',')]
    matrix_concurrency: Option<Vec<usize>>,

    /// Prepend a synthetic ~N-token system prompt to every request, identical across requests.
    /// Used to exercise server-side prefix caching.
    #[arg(long, value_name = "N", conflicts_with = "prefix_file")]
    prefix_tokens: Option<usize>,

    /// Load a system prompt from file and prepend it to every request.
    /// Used to exercise server-side prefix caching.
    #[arg(long, value_name = "PATH")]
    prefix_file: Option<PathBuf>,

    /// Replay multi-turn conversations from the dataset as a sequence of
    /// growing-prefix requests (turn 1, turns 1-2, turns 1-3, ...). Each
    /// request reuses the previous request's content as its leading tokens —
    /// exercises server-side prefix caching on a realistic chat workload.
    /// Requires a dataset with ShareGPT-style conversations.
    #[arg(long)]
    multi_turn: bool,

    /// Cap the number of turns replayed from each conversation (only applies
    /// with --multi-turn). Default: replay all turns up to --num-requests.
    #[arg(long, value_name = "N")]
    max_turns_per_conversation: Option<usize>,
}

const SYSTEM_PREFIX_TEMPLATE: &str = "\
You are a helpful and knowledgeable AI assistant. Answer questions clearly, accurately, and concisely. \
When explaining technical topics, use plain language and concrete examples grounded in real-world scenarios. \
If you do not know the answer to a question, say so directly rather than guessing or fabricating information. \
Format code in fenced markdown code blocks with the appropriate language tag. \
Format mathematical expressions using LaTeX notation when appropriate. \
Be honest about uncertainty in your answers, and acknowledge when a question is ambiguous or could have multiple interpretations. \
When asked to perform a task, do exactly what was requested without adding unsolicited improvements. \
Maintain a neutral, professional tone in your responses. Cite sources when you reference specific facts, papers, or external information. \
When discussing contested or controversial topics, present multiple perspectives fairly without advocating for a particular position. \
Prefer concrete examples over abstract explanations when teaching new concepts. \
Ask clarifying questions if the request is unclear or could be interpreted in more than one reasonable way. \
Respect the user's time by being direct and avoiding unnecessary preamble. \
When summarizing, preserve the original meaning and avoid introducing claims not supported by the source. \
For step-by-step reasoning tasks, lay out your work explicitly so the reader can verify each step. \
Prefer plain text over markdown when the response is going to be read in a terminal. \
Avoid emojis and decorative formatting unless they materially aid understanding. \
";

fn generate_system_prefix(target_tokens: usize) -> String {
    // chars/4 heuristic to roughly hit the token target.
    let target_chars = target_tokens.saturating_mul(4);
    let mut out = String::with_capacity(target_chars + SYSTEM_PREFIX_TEMPLATE.len());
    while out.len() < target_chars {
        out.push_str(SYSTEM_PREFIX_TEMPLATE);
    }
    // Truncate at a UTF-8 char boundary. Template is ASCII so any byte index works,
    // but be defensive in case it's edited later.
    while !out.is_char_boundary(target_chars.min(out.len())) {
        out.pop();
    }
    out.truncate(target_chars);
    out
}

/// One of the four ways we produce a list of chat requests:
///   - `Ready`: messages already constructed (real dataset prompts, or
///     multi-turn replay) — calibration just samples from the first one.
///   - `Pad`: corpus loaded but not yet sized — we calibrate first, then
///     concatenate corpus entries to hit `input_tokens × chars_per_token`.
enum Source {
    Ready(Vec<serde_json::Value>),
    Pad { corpus: Vec<String>, input_tokens: usize },
}

fn first_user_content(msgs: &[serde_json::Value]) -> Option<String> {
    msgs.first()
        .and_then(|m| m.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        })
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(String::from)
}

/// Resolve the request source from task config + flags.
/// Priority: explicit --multi-turn > explicit dataset > input_tokens (pad) > default sharegpt multi-turn.
async fn resolve_source(
    task: &TaskConfig,
    multi_turn_flag: bool,
    max_turns: Option<usize>,
    num_requests: usize,
    quiet: bool,
) -> Result<Source> {
    if multi_turn_flag {
        let dataset = task.dataset.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--multi-turn requires a dataset (set `dataset:` in the task or pass --dataset)")
        })?;
        if !quiet {
            println!("Loading multi-turn conversations from {dataset}...");
        }
        let msgs = load_multi_turn(dataset, num_requests, max_turns).await?;
        if !quiet {
            println!("Loaded {} growing-prefix requests", msgs.len());
        }
        Ok(Source::Ready(msgs))
    } else if let Some(dataset) = &task.dataset {
        if !quiet {
            println!("Loading dataset {dataset}...");
        }
        let loaded = load_dataset(dataset, num_requests, task.shuffle, task.seed).await?;
        if !quiet {
            println!("Loaded {} prompts (seed={})", loaded.len(), task.seed);
        }
        Ok(Source::Ready(prompts_to_messages(
            loaded.into_iter().map(|p| p.text).collect(),
        )))
    } else if task.input_tokens > 0 {
        if !quiet {
            println!(
                "Padding prompts from {DEFAULT_CORPUS} to ~{} tokens each...",
                task.input_tokens
            );
        }
        let corpus = load_corpus_for_padding(DEFAULT_CORPUS, num_requests, task.seed).await?;
        if !quiet {
            println!("Loaded corpus of {} entries", corpus.len());
        }
        Ok(Source::Pad {
            corpus,
            input_tokens: task.input_tokens,
        })
    } else {
        if !quiet {
            println!("Using {DEFAULT_CORPUS} multi-turn conversations (default)...");
        }
        let msgs = load_multi_turn(DEFAULT_CORPUS, num_requests, max_turns).await?;
        if !quiet {
            println!("Loaded {} growing-prefix requests", msgs.len());
        }
        Ok(Source::Ready(msgs))
    }
}

fn build_messages(source: Source, num_requests: usize, ratio: f64) -> Vec<serde_json::Value> {
    match source {
        Source::Ready(msgs) => msgs,
        Source::Pad {
            corpus,
            input_tokens,
        } => {
            let target_chars = (input_tokens as f64 * ratio) as usize;
            let prompts = pad_to_target_chars(&corpus, num_requests, target_chars);
            prompts_to_messages(prompts)
        }
    }
}

/// Probe the server for its chars-per-token ratio. Falls back to 4.0 on any
/// failure and prints a one-line note so the user knows whether calibration
/// succeeded.
async fn calibrate_with_fallback(
    client: &OpenAIClient,
    sample: &str,
    model: Option<&str>,
    quiet: bool,
) -> f64 {
    // Cap probe length — a few hundred chars is plenty for a stable ratio and
    // keeps the probe cheap on slow servers.
    let probe: String = sample.chars().take(500).collect();
    if probe.trim().is_empty() {
        if !quiet {
            println!("  Tokenizer:   default 4.0 chars/token (no probe sample available)");
        }
        return 4.0;
    }
    match client.calibrate_tokenizer(&probe, model).await {
        Ok(ratio) => {
            if !quiet {
                println!("  Tokenizer:   1 token ≈ {:.2} chars (calibrated)", ratio);
            }
            ratio
        }
        Err(e) => {
            if !quiet {
                println!(
                    "  Tokenizer:   default 4.0 chars/token (calibration failed: {})",
                    e.to_string().chars().take(80).collect::<String>()
                );
            }
            4.0
        }
    }
}

fn resolve_prefix(cli: &Cli) -> Result<Option<String>> {
    if let Some(n) = cli.prefix_tokens {
        if n == 0 {
            return Ok(None);
        }
        Ok(Some(generate_system_prefix(n)))
    } else if let Some(path) = &cli.prefix_file {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read prefix file {}", path.display()))?;
        if content.is_empty() {
            Ok(None)
        } else {
            Ok(Some(content))
        }
    } else {
        Ok(None)
    }
}

fn get_task_config(cli: &Cli) -> Result<(TaskConfig, Option<config::Config>)> {
    let config_path = cli.config.clone().or_else(|| find_config());
    let config = config_path
        .as_ref()
        .map(|p| load_config(p))
        .transpose()?;

    if let Some(task_name) = &cli.task {
        let cfg = config.ok_or_else(|| anyhow::anyhow!("No config file found"))?;
        let mut task = cfg
            .tasks
            .get(task_name)
            .cloned()
            .ok_or_else(|| {
                let names: Vec<_> = cfg.tasks.keys().cloned().collect();
                anyhow::anyhow!(
                    "Task '{}' not found. Available: {}",
                    task_name,
                    names.join(", ")
                )
            })?;

        // CLI overrides
        if let Some(v) = &cli.base_url { task.base_url = v.clone(); }
        if let Some(v) = &cli.model { task.model = Some(v.clone()); }
        if let Some(v) = &cli.token { task.token = Some(v.clone()); }
        if let Some(v) = &cli.api_key { task.api_key = Some(v.clone()); }
        if let Some(v) = cli.num_requests { task.num_requests = v; }
        if let Some(v) = cli.concurrency { task.concurrency = v; }
        if let Some(v) = cli.max_tokens { task.max_tokens = v; }
        if let Some(v) = cli.input_tokens { task.input_tokens = v; }
        if let Some(v) = cli.timeout { task.timeout = v; }
        if cli.no_verify_ssl { task.verify_ssl = false; }
        if let Some(v) = &cli.dataset { task.dataset = Some(v.clone()); }
        if cli.no_shuffle { task.shuffle = false; }
        task.seed = cli.seed;
        if cli.warmup > 0 { task.warmup = cli.warmup; }
        if cli.retries > 0 { task.retries = cli.retries; }

        return Ok((task, Some(cfg)));
    }

    // No task name — use CLI args directly
    let base_url = cli
        .base_url
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--base-url required (or use a task name from config)"))?;

    Ok((TaskConfig {
        name: "cli".to_string(),
        base_url,
        model: cli.model.clone(),
        auth: None,
        token: cli.token.clone(),
        api_key: cli.api_key.clone(),
        num_requests: cli.num_requests.unwrap_or(10),
        concurrency: cli.concurrency.unwrap_or(1),
        max_tokens: cli.max_tokens.unwrap_or(256),
        input_tokens: cli.input_tokens.unwrap_or(0),
        timeout: cli.timeout.unwrap_or(120.0),
        verify_ssl: !cli.no_verify_ssl,
        no_verify_ssl: None,
        dataset: cli.dataset.clone(),
        shuffle: !cli.no_shuffle,
        seed: cli.seed,
        warmup: cli.warmup,
        retries: cli.retries,
    }, None))
}

fn print_summary(result: &BenchmarkResult, task: &TaskConfig) {
    let bold = Style::new().bold();
    let green = Style::new().green();
    let cyan = Style::new().cyan();
    let yellow = Style::new().yellow();

    println!();
    println!("{}", bold.apply_to("Benchmark Results"));
    println!("{}", "=".repeat(50));

    let errors_str = if result.num_errors > 0 {
        yellow.apply_to(result.num_errors.to_string()).to_string()
    } else {
        green.apply_to(result.num_errors.to_string()).to_string()
    };
    println!(
        "Requests:  {}/{} completed  ({} errors)",
        green.apply_to(result.num_completed),
        result.num_requests,
        errors_str,
    );
    println!("Duration:  {:.2}s", result.total_duration);
    println!();

    println!("{}", bold.apply_to("Throughput:"));
    println!("  Requests/sec  {}", cyan.apply_to(format!("{:.2}", result.requests_per_second)));
    println!("  Output TPS    {}", cyan.apply_to(format!("{:.2}", result.output_tps)));
    println!("  Prefill TPS   {}", cyan.apply_to(format!("{:.2}", result.prefill_tps_mean)));
    println!();

    println!("{}", bold.apply_to("Latency (seconds):"));
    println!(
        "  TTFT    mean={:.3}  p50={:.3}  p75={:.3}  p90={:.3}  p95={:.3}  p98={:.3}  p99={:.3}",
        result.ttft_mean,
        result.ttft_p50,
        result.ttft_p75,
        result.ttft_p90,
        result.ttft_p95,
        result.ttft_p98,
        result.ttft_p99,
    );
    println!(
        "  E2E     mean={:.3}  p50={:.3}  p95={:.3}  p99={:.3}",
        result.e2e_mean, result.e2e_p50, result.e2e_p95, result.e2e_p99,
    );
    println!("  ITL     mean={:.4}", result.inter_token_latency_mean);
    println!();

    println!("{}", bold.apply_to("Tokens:"));
    let n = result.num_completed.max(1);
    let input_suffix = if result.input_tokens_estimated_count > 0 {
        format!(
            ", {} estimated",
            yellow.apply_to(result.input_tokens_estimated_count)
        )
    } else {
        String::new()
    };
    let output_suffix = if result.output_tokens_estimated_count > 0 {
        format!(
            ", {} estimated",
            yellow.apply_to(result.output_tokens_estimated_count)
        )
    } else {
        String::new()
    };
    println!(
        "  Input   {} total  (mean: {:.1}{})",
        result.total_input_tokens,
        result.total_input_tokens as f64 / n as f64,
        input_suffix,
    );
    println!(
        "  Output  {} total  (mean: {:.1}{})",
        result.total_output_tokens,
        result.total_output_tokens as f64 / n as f64,
        output_suffix,
    );

    if !result.errors.is_empty() {
        println!();
        println!("{}", yellow.apply_to(format!("Errors ({}):", result.errors.len())));
        for e in result.errors.iter().take(5) {
            let truncated: String = e.chars().take(80).collect();
            println!("  - {}", truncated);
        }
        if result.errors.len() > 5 {
            println!("  ... and {} more", result.errors.len() - 5);
        }
    }

    let _ = task; // available for future use (e.g., printing task name)
}

fn result_to_json(result: &BenchmarkResult) -> serde_json::Value {
    serde_json::json!({
        "num_requests": result.num_requests,
        "num_completed": result.num_completed,
        "num_errors": result.num_errors,
        "total_duration": result.total_duration,
        "throughput": {
            "requests_per_second": result.requests_per_second,
            "output_tps": result.output_tps,
            "prefill_tps": result.prefill_tps_mean,
        },
        "latency": {
            "ttft": {
                "mean": result.ttft_mean,
                "p50": result.ttft_p50,
                "p75": result.ttft_p75,
                "p90": result.ttft_p90,
                "p95": result.ttft_p95,
                "p98": result.ttft_p98,
                "p99": result.ttft_p99,
            },
            "e2e": {
                "mean": result.e2e_mean,
                "p50": result.e2e_p50,
                "p95": result.e2e_p95,
                "p99": result.e2e_p99,
            },
            "inter_token_mean": result.inter_token_latency_mean,
        },
        "tokens": {
            "input": result.total_input_tokens,
            "output": result.total_output_tokens,
            "input_estimated_count": result.input_tokens_estimated_count,
            "output_estimated_count": result.output_tokens_estimated_count,
        },
        "errors": result.errors,
    })
}

fn write_csv(path: &PathBuf, result: &BenchmarkResult, task: &TaskConfig) -> Result<()> {
    use std::io::Write;

    let file_exists = path.exists();
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);

    let headers = [
        "task", "dataset", "concurrency", "num_requests", "num_completed", "num_errors",
        "total_duration", "requests_per_second", "output_tps", "prefill_tps",
        "ttft_mean", "ttft_p50", "ttft_p75", "ttft_p90", "ttft_p95", "ttft_p98", "ttft_p99",
        "e2e_mean", "e2e_p50", "e2e_p95", "e2e_p99",
        "inter_token_latency_mean",
        "total_input_tokens", "total_output_tokens",
        "input_tokens_estimated", "output_tokens_estimated",
    ];

    if !file_exists {
        writeln!(writer, "{}", headers.join(","))?;
    }

    writeln!(
        writer,
        "{},{},{},{},{},{},{:.6},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{}",
        task.name,
        task.dataset.as_deref().unwrap_or(""),
        task.concurrency,
        result.num_requests,
        result.num_completed,
        result.num_errors,
        result.total_duration,
        result.requests_per_second,
        result.output_tps,
        result.prefill_tps_mean,
        result.ttft_mean,
        result.ttft_p50,
        result.ttft_p75,
        result.ttft_p90,
        result.ttft_p95,
        result.ttft_p98,
        result.ttft_p99,
        result.e2e_mean,
        result.e2e_p50,
        result.e2e_p95,
        result.e2e_p99,
        result.inter_token_latency_mean,
        result.total_input_tokens,
        result.total_output_tokens,
        result.input_tokens_estimated_count,
        result.output_tokens_estimated_count,
    )?;

    Ok(())
}

async fn run_task(
    task: &TaskConfig,
    output: Option<&str>,
    quiet: bool,
    debug: bool,
    prefix: Option<String>,
    multi_turn: bool,
    max_turns: Option<usize>,
) -> Result<i32> {
    let mut client = OpenAIClient::new(
        task.base_url.clone(),
        task.token.clone(),
        task.api_key.clone(),
        task.timeout,
        task.effective_verify_ssl(),
    )?;
    client.debug = debug;
    client.system_prefix = prefix.clone();

    if !quiet {
        println!("yabench — Running task: {}", task.name);
        println!("  URL:         {}", task.base_url);
        println!("  Model:       {}", task.model.as_deref().unwrap_or("(default)"));
        println!(
            "  Requests:    {} (concurrency: {})",
            task.num_requests, task.concurrency
        );
        println!("  Max tokens:  {}", task.max_tokens);
        if let Some(ds) = &task.dataset {
            println!("  Dataset:     {ds}");
        }
        if task.warmup > 0 {
            println!("  Warmup:      {} requests", task.warmup);
        }
        if let Some(p) = &prefix {
            let approx_tokens = (p.chars().count() / 4).max(1);
            println!("  Prefix:      ~{approx_tokens} tokens (system message)");
        }
        println!();
    }

    let source = resolve_source(task, multi_turn, max_turns, task.num_requests, quiet).await?;

    let calibration_sample: String = match &source {
        Source::Ready(msgs) => first_user_content(msgs)
            .unwrap_or_else(|| SYSTEM_PREFIX_TEMPLATE.to_string()),
        Source::Pad { corpus, .. } => corpus
            .first()
            .cloned()
            .unwrap_or_else(|| SYSTEM_PREFIX_TEMPLATE.to_string()),
    };
    let ratio =
        calibrate_with_fallback(&client, &calibration_sample, task.model.as_deref(), quiet).await;
    client.chars_per_token = ratio;
    if !quiet {
        println!();
    }

    let messages_list = build_messages(source, task.num_requests, ratio);

    let client = Arc::new(client);

    let num_requests = messages_list.len();
    let pb = if !quiet {
        let bar = ProgressBar::new(num_requests as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[{pos}/{len}] {msg}")
                .unwrap(),
        );
        Some(bar)
    } else {
        None
    };

    let bench_config = BenchmarkConfig {
        model: task.model.clone(),
        max_tokens: task.max_tokens,
        concurrency: task.concurrency,
        warmup: task.warmup,
        max_retries: task.retries,
    };

    let result = run_benchmark(client, messages_list, bench_config, pb).await;

    if !quiet {
        print_summary(&result, task);
    }

    if let Some(out) = output {
        let out_path = PathBuf::from(out);
        let ext = out_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if ext == "csv" {
            write_csv(&out_path, &result, task)?;
        } else {
            let json = result_to_json(&result);
            std::fs::write(&out_path, serde_json::to_string_pretty(&json)?)
                .with_context(|| format!("Failed to write {}", out_path.display()))?;
        }

        if !quiet {
            println!("\nResults saved to {out}");
        }
    }

    Ok(if result.num_errors == 0 { 0 } else { 1 })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle --chat
    if let Some(ref prompt) = cli.chat {
        let (mut task, cfg) = get_task_config(&cli)?;
        let auth_provider = task.auth.as_ref().and_then(|n| cfg.as_ref()?.auth.get(n));
        if let Some(AuthProvider::Token { token }) = auth_provider {
            task.token = Some(token.clone());
        } else if let Some(AuthProvider::ApiKey { api_key }) = auth_provider {
            task.api_key = Some(api_key.clone());
        } else if let Some(AuthProvider::Iam { iam_url, domain, username, password, project }) = auth_provider {
            let t = iam::fetch_token(iam_url, domain, username, password, project, task.effective_verify_ssl()).await?;
            task.token = Some(t.token);
        }
        let mut client = OpenAIClient::new(
            task.base_url.clone(),
            task.token.clone(),
            task.api_key.clone(),
            task.timeout,
            task.effective_verify_ssl(),
        )?;
        client.debug = cli.debug;
        client.system_prefix = resolve_prefix(&cli)?;
        let (_, metrics) = client.chat(prompt, task.model.as_deref(), task.max_tokens.max(512)).await?;
        eprintln!("\n[TTFT: {:.3}s | tokens: {} | {:.2}s total]",
            metrics.ttft.unwrap_or(0.0),
            metrics.output_tokens,
            metrics.total_time,
        );
        if let Some(err) = metrics.error {
            anyhow::bail!(err);
        }
        return Ok(());
    }

    // Handle --init
    if cli.init {
        let path = std::path::Path::new("yabench.yaml");
        if path.exists() {
            anyhow::bail!("yabench.yaml already exists");
        }
        std::fs::write(path, include_str!("../yabench.example.yaml"))?;
        println!("Created yabench.yaml — edit it to configure your tasks.");
        return Ok(());
    }

    // Handle --guide
    if cli.guide {
        print!("{}", include_str!("../docs/guide.md"));
        return Ok(());
    }

    // Handle --list-datasets
    if cli.list_datasets {
        use datasets::{EMBEDDED_DATASETS, REMOTE_DATASETS};
        println!("Embedded (built-in):");
        for (name, bytes) in EMBEDDED_DATASETS {
            let lines = bytes.iter().filter(|&&b| b == b'\n').count();
            println!("  {name:<20} ({lines} prompts)");
        }
        println!("\nRemote (downloaded on demand):");
        for (name, url) in REMOTE_DATASETS {
            println!("  {name:<20} {url}");
        }
        return Ok(());
    }

    // Handle --print-dataset
    if let Some(ref name) = cli.print_dataset {
        let prompts = load_dataset(name, cli.num_requests.unwrap_or(10), !cli.no_shuffle, cli.seed).await?;
        for (i, p) in prompts.iter().enumerate() {
            println!("[{}] {}", i + 1, p.text);
            println!();
        }
        return Ok(());
    }

    // Handle --download
    if let Some(ref dataset) = cli.download {
        println!("Downloading datasets...");
        if dataset == "all" {
            download_all_datasets(false).await;
        } else {
            download_builtin(dataset, false).await?;
        }
        println!("\nDone. Datasets cached in ~/.cache/yabench/datasets/");
        return Ok(());
    }

    // Handle --list
    if cli.list {
        let config_path = cli.config.clone().or_else(|| find_config());
        let config_path = config_path.ok_or_else(|| anyhow::anyhow!("No config file found"))?;
        let config = load_config(&config_path)?;
        println!("Tasks in {}:", config_path.display());
        for (name, task) in &config.tasks {
            println!("  {}: {}", name, task.base_url);
        }
        return Ok(());
    }

    let (mut task, cfg) = get_task_config(&cli)?;

    // Resolve auth provider
    let auth_provider = task.auth.as_ref().and_then(|name| {
        cfg.as_ref()?.auth.get(name)
    });

    match auth_provider {
        Some(AuthProvider::Token { token }) => {
            task.token = Some(token.clone());
        }
        Some(AuthProvider::ApiKey { api_key }) => {
            task.api_key = Some(api_key.clone());
        }
        Some(AuthProvider::Iam { iam_url, domain, username, password, project }) => {
            if !cli.quiet {
                println!("Fetching IAM token for {}@{}...", username, project);
            }
            let t = iam::fetch_token(iam_url, domain, username, password, project, task.effective_verify_ssl()).await?;
            if !cli.quiet {
                println!("Token obtained (expires: {})", t.expires_at);
                println!();
            }
            task.token = Some(t.token);
        }
        None => {}
    }

    if let Some(report_path) = cli.perf_report.as_ref() {
        let suite_cfg = perf::PerfSuiteConfig::default();
        let prefix = resolve_prefix(&cli)?;

        let mut client = OpenAIClient::new(
            task.base_url.clone(),
            task.token.clone(),
            task.api_key.clone(),
            task.timeout,
            task.effective_verify_ssl(),
        )?;
        client.debug = cli.debug;
        client.system_prefix = prefix.clone();

        let source = resolve_source(
            &task,
            cli.multi_turn,
            cli.max_turns_per_conversation,
            suite_cfg.n_per_level,
            cli.quiet,
        )
        .await?;

        let calibration_sample: String = match &source {
            Source::Ready(msgs) => first_user_content(msgs)
                .unwrap_or_else(|| SYSTEM_PREFIX_TEMPLATE.to_string()),
            Source::Pad { corpus, .. } => corpus
                .first()
                .cloned()
                .unwrap_or_else(|| SYSTEM_PREFIX_TEMPLATE.to_string()),
        };
        let ratio = calibrate_with_fallback(
            &client,
            &calibration_sample,
            task.model.as_deref(),
            cli.quiet,
        )
        .await;
        client.chars_per_token = ratio;

        let messages_list = build_messages(source, suite_cfg.n_per_level, ratio);
        let client = Arc::new(client);

        let final_path = if report_path.is_empty() {
            format!(
                "perf-report-{}.md",
                perf::format_filename_timestamp(perf::now_secs())
            )
        } else {
            report_path.clone()
        };

        if !cli.quiet {
            println!("yabench perf suite");
            println!("  task:        {}", task.name);
            println!("  endpoint:    {}", task.base_url);
            println!("  concurrency: {:?}", suite_cfg.concurrencies);
            println!("  per-level n: {}", suite_cfg.n_per_level);
            if let Some(p) = &prefix {
                let approx_tokens = (p.chars().count() / 4).max(1);
                println!("  prefix:      ~{approx_tokens} tokens (system message)");
            }
            println!("  output:      {}", final_path);
            println!();
        }

        let report =
            perf::run_perf_suite(client, messages_list, &task, &suite_cfg, cli.quiet).await?;

        let md = perf::format_markdown(&report);
        std::fs::write(&final_path, &md)
            .with_context(|| format!("Failed to write {}", final_path))?;

        if !cli.quiet {
            println!("\nReport written to {}", final_path);
        }

        return Ok(());
    }

    if let Some(matrix_path) = cli.perf_matrix.as_ref() {
        let prefix = resolve_prefix(&cli)?;

        let mut client = OpenAIClient::new(
            task.base_url.clone(),
            task.token.clone(),
            task.api_key.clone(),
            task.timeout,
            task.effective_verify_ssl(),
        )?;
        client.debug = cli.debug;
        client.system_prefix = prefix.clone();

        let mut matrix_cfg = perf::MatrixConfig {
            n_per_cell: cli.matrix_n,
            ..Default::default()
        };
        if let Some(ref v) = cli.matrix_input { matrix_cfg.input_tokens = v.clone(); }
        if let Some(ref v) = cli.matrix_output { matrix_cfg.output_tokens = v.clone(); }
        if let Some(ref v) = cli.matrix_concurrency { matrix_cfg.concurrencies = v.clone(); }

        let corpus = load_corpus_for_padding(DEFAULT_CORPUS, matrix_cfg.n_per_cell, task.seed).await?;

        let sample = corpus.first().cloned().unwrap_or_else(|| SYSTEM_PREFIX_TEMPLATE.to_string());
        let ratio = calibrate_with_fallback(&client, &sample, task.model.as_deref(), cli.quiet).await;
        client.chars_per_token = ratio;

        let client = Arc::new(client);

        let final_path = if matrix_path.is_empty() {
            format!(
                "perf-matrix-{}.md",
                perf::format_filename_timestamp(perf::now_secs())
            )
        } else {
            matrix_path.clone()
        };

        let total_cells = matrix_cfg.input_tokens.len()
            * matrix_cfg.output_tokens.len()
            * matrix_cfg.concurrencies.len();

        if !cli.quiet {
            println!("yabench perf matrix");
            println!("  task:        {}", task.name);
            println!("  endpoint:    {}", task.base_url);
            println!("  grid:        {} input × {} output × {} concurrency = {} cells",
                matrix_cfg.input_tokens.len(),
                matrix_cfg.output_tokens.len(),
                matrix_cfg.concurrencies.len(),
                total_cells,
            );
            println!("  per-cell n:  {}", matrix_cfg.n_per_cell);
            if let Some(p) = &prefix {
                let approx_tokens = (p.chars().count() / 4).max(1);
                println!("  prefix:      ~{approx_tokens} tokens (system message)");
            }
            println!("  output:      {}", final_path);
            println!();
        }

        let report = perf::run_perf_matrix(client, &corpus, &task, &matrix_cfg, cli.quiet).await?;

        let md = perf::format_matrix_markdown(&report);
        std::fs::write(&final_path, &md)
            .with_context(|| format!("Failed to write {}", final_path))?;

        if !cli.quiet {
            println!("\nMatrix report written to {}", final_path);
        }

        return Ok(());
    }

    let prefix = resolve_prefix(&cli)?;
    let exit_code = run_task(
        &task,
        cli.output.as_deref(),
        cli.quiet,
        cli.debug,
        prefix,
        cli.multi_turn,
        cli.max_turns_per_conversation,
    )
    .await?;

    std::process::exit(exit_code);
}
