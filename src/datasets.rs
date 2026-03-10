use anyhow::{anyhow, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Prompt {
    pub text: String,
    pub system: Option<String>,
}

// Local datasets embedded directly into the binary at compile time.
pub const EMBEDDED_DATASETS: &[(&str, &[u8])] = &[
    ("sample-en", include_bytes!("../datasets/sample-en.jsonl")),
    ("sample-zh", include_bytes!("../datasets/sample-zh.jsonl")),
    ("alpaca-en", include_bytes!("../datasets/alpaca-en-500.jsonl")),
    ("alpaca-zh", include_bytes!("../datasets/alpaca-zh-500.jsonl")),
];

const LOCAL_DATASETS: &[(&str, &str)] = &[
    ("sample-en", "sample-en.jsonl"),
    ("sample-zh", "sample-zh.jsonl"),
    ("alpaca-en", "alpaca-en-500.jsonl"),
    ("alpaca-zh", "alpaca-zh-500.jsonl"),
];

pub const REMOTE_DATASETS: &[(&str, &str)] = &[
    (
        "sharegpt",
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
    ),
    (
        "sharegpt-small",
        "https://huggingface.co/datasets/shibing624/sharegpt_gpt4/resolve/main/sharegpt_gpt4.jsonl",
    ),
    (
        "belle",
        "https://huggingface.co/datasets/BelleGroup/train_1M_CN/resolve/main/Belle_open_source_1M.json",
    ),
    (
        "firefly",
        "https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M/resolve/main/firefly-train-1.1M.jsonl",
    ),
];

pub fn builtin_dataset_names() -> Vec<&'static str> {
    LOCAL_DATASETS
        .iter()
        .chain(REMOTE_DATASETS.iter())
        .map(|(name, _)| *name)
        .collect()
}

pub fn cache_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("yabench")
        .join("datasets")
}

fn embedded_dataset(name: &str) -> Option<&'static [u8]> {
    EMBEDDED_DATASETS.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
}

fn get_remote_cache_path(name: &str) -> Option<PathBuf> {
    for (n, url) in REMOTE_DATASETS {
        if *n == name {
            let ext = if url.ends_with(".jsonl") { ".jsonl" } else { ".json" };
            return Some(cache_dir().join(format!("{name}{ext}")));
        }
    }
    None
}

fn parse_item(item: &Value) -> Option<Prompt> {
    let mut text = None;
    let mut system = None;

    if let Some(prompt) = item.get("prompt").and_then(|v| v.as_str()) {
        text = Some(prompt.to_string());
    } else if let Some(instruction) = item.get("instruction").and_then(|v| v.as_str()) {
        let input = item
            .get("input")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if input.is_empty() {
            text = Some(instruction.trim().to_string());
        } else {
            text = Some(format!("{}\n\n{}", instruction.trim(), input));
        }
    } else if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
        text = Some(t.to_string());
    } else if let Some(c) = item.get("content").and_then(|v| v.as_str()) {
        text = Some(c.to_string());
    } else if let Some(q) = item.get("question").and_then(|v| v.as_str()) {
        text = Some(q.to_string());
    } else if let Some(msgs) = item.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if role == "system" {
                system = Some(content.to_string());
            } else if role == "user" && text.is_none() {
                text = Some(content.to_string());
            }
        }
    } else if let Some(convs) = item.get("conversations").and_then(|v| v.as_array()) {
        for msg in convs {
            let from = msg.get("from").and_then(|v| v.as_str()).unwrap_or("");
            if from == "human" {
                text = msg.get("value").and_then(|v| v.as_str()).map(String::from);
                break;
            }
        }
    }

    let text = text?.trim().to_string();
    if text.len() <= 10 {
        return None;
    }

    Some(Prompt { text, system })
}

pub fn load_jsonl(path: &Path, max_prompts: Option<usize>) -> Result<Vec<Prompt>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let mut prompts = vec![];

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let item: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(p) = parse_item(&item) {
            prompts.push(p);
        }
        if let Some(max) = max_prompts {
            if prompts.len() >= max {
                break;
            }
        }
    }

    Ok(prompts)
}

pub fn load_json(path: &Path, max_prompts: Option<usize>) -> Result<Vec<Prompt>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let data: Value = serde_json::from_str(&content)?;
    let items = data.as_array().ok_or_else(|| anyhow!("Expected JSON array"))?;

    let mut prompts = vec![];
    for item in items {
        if let Some(p) = parse_item(item) {
            prompts.push(p);
        }
        if let Some(max) = max_prompts {
            if prompts.len() >= max {
                break;
            }
        }
    }

    Ok(prompts)
}

pub fn load_txt(path: &Path, max_prompts: Option<usize>) -> Result<Vec<Prompt>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let mut prompts = vec![];

    for line in content.lines() {
        let text = line.trim();
        if text.len() > 10 {
            prompts.push(Prompt {
                text: text.to_string(),
                system: None,
            });
        }
        if let Some(max) = max_prompts {
            if prompts.len() >= max {
                break;
            }
        }
    }

    Ok(prompts)
}

pub async fn download_dataset(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    let response = client.get(url).send().await?;
    let status = response.status();
    if !status.is_success() {
        return Err(anyhow!("HTTP {}: {}", status, url));
    }

    let total = response.content_length();
    let pb = ProgressBar::new(total.unwrap_or(0));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut file = std::fs::File::create(dest)
        .with_context(|| format!("Failed to create {}", dest.display()))?;

    use std::io::Write;
    let mut stream = response.bytes_stream();
    use tokio_stream::StreamExt;
    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        pb.inc(bytes.len() as u64);
        file.write_all(&bytes)?;
    }

    pb.finish_and_clear();
    Ok(())
}

pub async fn download_builtin(name: &str, quiet: bool) -> Result<PathBuf> {
    // Local datasets are embedded — nothing to download
    if embedded_dataset(name).is_some() {
        if !quiet {
            println!("  {name}: embedded in binary");
        }
        return Ok(PathBuf::from(name)); // dummy path, not used
    }

    // Remote dataset
    for (n, url) in REMOTE_DATASETS {
        if *n == name {
            let ext = if url.ends_with(".jsonl") { ".jsonl" } else { ".json" };
            let dest = cache_dir().join(format!("{name}{ext}"));

            if dest.exists() {
                if !quiet {
                    println!("  {name}: already downloaded ({})", dest.display());
                }
                return Ok(dest);
            }

            if !quiet {
                println!("  {name}: downloading from {url}...");
            }
            download_dataset(url, &dest).await?;
            if !quiet {
                let size_mb = dest.metadata().map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
                println!("  {name}: done ({:.1} MB)", size_mb);
            }
            return Ok(dest);
        }
    }

    Err(anyhow!(
        "Unknown dataset: '{name}'. Available: {}",
        builtin_dataset_names().join(", ")
    ))
}

pub async fn download_all_datasets(quiet: bool) {
    for (name, _) in LOCAL_DATASETS.iter().chain(REMOTE_DATASETS.iter()) {
        match download_builtin(name, quiet).await {
            Ok(_) => {}
            Err(e) => {
                if !quiet {
                    eprintln!("  {name}: failed - {e}");
                }
            }
        }
    }
}

// Fisher-Yates shuffle using a simple LCG seeded RNG (no external dep)
fn seeded_shuffle<T>(data: &mut Vec<T>, seed: u64) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    // Splitmix64 RNG
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut next = move || -> u64 {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };

    for i in (1..n).rev() {
        let j = (next() % (i + 1) as u64) as usize;
        data.swap(i, j);
    }
}

fn load_jsonl_bytes(bytes: &[u8], max_prompts: Option<usize>) -> Result<Vec<Prompt>> {
    let text = std::str::from_utf8(bytes).context("embedded dataset is not valid UTF-8")?;
    let mut prompts = vec![];
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let item: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(p) = parse_item(&item) {
            prompts.push(p);
        }
        if let Some(max) = max_prompts {
            if prompts.len() >= max { break; }
        }
    }
    Ok(prompts)
}

pub async fn load_dataset(
    source: &str,
    num_prompts: usize,
    shuffle: bool,
    seed: u64,
) -> Result<Vec<Prompt>> {
    let max_load = if shuffle { num_prompts * 3 } else { num_prompts };
    let max_load = if max_load == 0 { None } else { Some(max_load) };

    let mut prompts = if let Some(bytes) = embedded_dataset(source) {
        // Local dataset embedded in binary
        load_jsonl_bytes(bytes, max_load)?
    } else if REMOTE_DATASETS.iter().any(|(n, _)| *n == source) {
        // Remote dataset — download to cache if needed
        let path = get_remote_cache_path(source).unwrap();
        if !path.exists() {
            println!("Downloading {source} dataset...");
            download_builtin(source, true).await?;
            println!("Saved to {}", path.display());
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if ext == "json" { load_json(&path, max_load)? } else { load_jsonl(&path, max_load)? }
    } else {
        // User-provided file path
        let path = PathBuf::from(source);
        if !path.exists() {
            return Err(anyhow!("Dataset not found: '{source}'"));
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        match ext.as_str() {
            "json" => load_json(&path, max_load)?,
            "jsonl" => load_jsonl(&path, max_load)?,
            "txt" => load_txt(&path, max_load)?,
            _ => load_jsonl(&path, max_load).or_else(|_| load_txt(&path, max_load))?,
        }
    };

    if prompts.is_empty() {
        return Err(anyhow!("No valid prompts found in '{source}'"));
    }

    if shuffle {
        seeded_shuffle(&mut prompts, seed);
    }

    prompts.truncate(num_prompts);
    Ok(prompts)
}
