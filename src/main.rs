mod benchmark;
mod client;
mod config;
mod datasets;
mod iam;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use console::Style;
use indicatif::{ProgressBar, ProgressStyle};

use benchmark::{run_benchmark, BenchmarkConfig, BenchmarkResult, generate_prompts};
use client::OpenAIClient;
use config::{find_config, load_config, AuthProvider, TaskConfig};
use datasets::{download_all_datasets, download_builtin, load_dataset};

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

    /// Max tokens per response
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Approximate input tokens per prompt
    #[arg(long)]
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

    /// Number of warmup requests
    #[arg(long, default_value = "0")]
    warmup: usize,

    /// Max retries on error
    #[arg(long, default_value = "0")]
    retries: u32,

    /// Send a single chat message and stream the response (e.g. --chat "hello")
    #[arg(long, value_name = "PROMPT")]
    chat: Option<String>,
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
        input_tokens: cli.input_tokens.unwrap_or(100),
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
    println!(
        "  Input   {} total  (mean: {:.1})",
        result.total_input_tokens,
        result.total_input_tokens as f64 / n as f64
    );
    println!(
        "  Output  {} total  (mean: {:.1})",
        result.total_output_tokens,
        result.total_output_tokens as f64 / n as f64
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
    ];

    if !file_exists {
        writeln!(writer, "{}", headers.join(","))?;
    }

    writeln!(
        writer,
        "{},{},{},{},{},{},{:.6},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
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
    )?;

    Ok(())
}

async fn run_task(task: &TaskConfig, output: Option<&str>, quiet: bool, debug: bool) -> Result<i32> {
    let mut client = OpenAIClient::new(
        task.base_url.clone(),
        task.token.clone(),
        task.api_key.clone(),
        task.timeout,
        task.effective_verify_ssl(),
    )?;
    client.debug = debug;
    let client = Arc::new(client);

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
        println!();
    }

    // Load prompts
    let prompts: Vec<String> = if let Some(dataset) = &task.dataset {
        if !quiet {
            println!("Loading dataset...");
        }
        let loaded = load_dataset(dataset, task.num_requests, task.shuffle, task.seed).await?;
        if !quiet {
            println!("Loaded {} prompts (seed={})", loaded.len(), task.seed);
            println!();
        }
        loaded.into_iter().map(|p| p.text).collect()
    } else {
        generate_prompts(task.num_requests, task.input_tokens)
    };

    let pb = if !quiet {
        let bar = ProgressBar::new(task.num_requests as u64);
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

    let result = run_benchmark(client, prompts, bench_config, pb).await;

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

    let exit_code = run_task(&task, cli.output.as_deref(), cli.quiet, cli.debug).await?;

    std::process::exit(exit_code);
}
