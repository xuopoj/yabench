use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Parse a token count with optional SI suffix: "100", "4K", "128K", "1.5M".
/// Case-insensitive. K = 1_000, M = 1_000_000 (decimal, matches how context
/// windows are marketed).
pub fn parse_size(s: &str) -> Result<usize, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty size".to_string());
    }

    let (num_part, mult): (&str, u64) = if let Some(stripped) = s
        .strip_suffix('K')
        .or_else(|| s.strip_suffix('k'))
    {
        (stripped, 1_000)
    } else if let Some(stripped) = s.strip_suffix('M').or_else(|| s.strip_suffix('m')) {
        (stripped, 1_000_000)
    } else {
        (s, 1)
    };

    let num: f64 = num_part
        .trim()
        .parse()
        .map_err(|_| format!("invalid size: '{s}'"))?;

    if !num.is_finite() || num < 0.0 {
        return Err(format!("size must be non-negative: '{s}'"));
    }

    Ok((num * mult as f64) as usize)
}

pub fn parse_size_u32(s: &str) -> Result<u32, String> {
    let v = parse_size(s)?;
    u32::try_from(v).map_err(|_| format!("size '{s}' exceeds u32::MAX"))
}

fn deserialize_size_usize<'de, D: Deserializer<'de>>(d: D) -> Result<usize, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Num(u64),
        Str(String),
    }
    match Repr::deserialize(d)? {
        Repr::Num(n) => Ok(n as usize),
        Repr::Str(s) => parse_size(&s).map_err(serde::de::Error::custom),
    }
}

fn deserialize_size_u32<'de, D: Deserializer<'de>>(d: D) -> Result<u32, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Num(u64),
        Str(String),
    }
    match Repr::deserialize(d)? {
        Repr::Num(n) => u32::try_from(n).map_err(|_| serde::de::Error::custom("value exceeds u32::MAX")),
        Repr::Str(s) => parse_size_u32(&s).map_err(serde::de::Error::custom),
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthProvider {
    Token {
        token: String,
    },
    ApiKey {
        api_key: String,
    },
    Iam {
        iam_url: String,
        domain: String,
        username: String,
        password: String,
        project: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TaskConfig {
    pub name: String,
    pub base_url: String,
    pub model: Option<String>,
    pub auth: Option<String>, // references a key in auth map
    // inline auth (for backwards compat / CLI)
    pub token: Option<String>,
    pub api_key: Option<String>,
    pub num_requests: usize,
    pub concurrency: usize,
    #[serde(deserialize_with = "deserialize_size_u32")]
    pub max_tokens: u32,
    #[serde(deserialize_with = "deserialize_size_usize")]
    pub input_tokens: usize,
    pub timeout: f64,
    pub verify_ssl: bool,
    pub no_verify_ssl: Option<bool>,
    pub dataset: Option<String>,
    pub shuffle: bool,
    pub seed: u64,
    pub warmup: usize,
    pub retries: u32,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_url: String::new(),
            model: None,
            auth: None,
            token: None,
            api_key: None,
            num_requests: 10,
            concurrency: 1,
            max_tokens: 256,
            // 0 means "no explicit input size" — falls back to sharegpt
            // multi-turn replay. Set to a positive number (e.g. 128K) to pad
            // sharegpt content to that exact token count instead.
            input_tokens: 0,
            timeout: 120.0,
            verify_ssl: false,
            no_verify_ssl: None,
            dataset: None,
            shuffle: true,
            seed: 42,
            warmup: 0,
            retries: 0,
        }
    }
}

impl TaskConfig {
    pub fn effective_verify_ssl(&self) -> bool {
        if let Some(no_verify) = self.no_verify_ssl {
            return !no_verify;
        }
        self.verify_ssl
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    defaults: HashMap<String, serde_yaml::Value>,
    #[serde(default)]
    auth: HashMap<String, serde_yaml::Value>,
    #[serde(default)]
    tasks: HashMap<String, serde_yaml::Value>,
}

pub struct Config {
    pub auth: HashMap<String, AuthProvider>,
    pub tasks: HashMap<String, TaskConfig>,
}

fn resolve_env_vars(val: &serde_yaml::Value) -> serde_yaml::Value {
    match val {
        serde_yaml::Value::String(s) => {
            if s.starts_with("${") && s.ends_with('}') {
                let var_name = &s[2..s.len() - 1];
                let resolved = std::env::var(var_name).unwrap_or_default();
                serde_yaml::Value::String(resolved)
            } else {
                val.clone()
            }
        }
        serde_yaml::Value::Mapping(m) => {
            let resolved: serde_yaml::Mapping = m
                .iter()
                .map(|(k, v)| (k.clone(), resolve_env_vars(v)))
                .collect();
            serde_yaml::Value::Mapping(resolved)
        }
        serde_yaml::Value::Sequence(seq) => {
            serde_yaml::Value::Sequence(seq.iter().map(resolve_env_vars).collect())
        }
        _ => val.clone(),
    }
}

fn merge_yaml(base: &serde_yaml::Value, overlay: &serde_yaml::Value) -> serde_yaml::Value {
    match (base, overlay) {
        (serde_yaml::Value::Mapping(b), serde_yaml::Value::Mapping(o)) => {
            let mut merged = b.clone();
            for (k, v) in o {
                merged.insert(k.clone(), v.clone());
            }
            serde_yaml::Value::Mapping(merged)
        }
        _ => overlay.clone(),
    }
}

pub fn load_config(path: &Path) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;

    let raw: RawConfig = serde_yaml::from_str(&content)
        .with_context(|| format!("Failed to parse config: {}", path.display()))?;

    // Parse auth providers
    let mut auth = HashMap::new();
    for (name, val) in raw.auth {
        let resolved = resolve_env_vars(&val);
        let provider: AuthProvider = serde_yaml::from_value(resolved)
            .with_context(|| format!("Failed to parse auth provider '{name}'"))?;
        auth.insert(name, provider);
    }

    let defaults_val = serde_yaml::to_value(&raw.defaults)
        .unwrap_or(serde_yaml::Value::Mapping(Default::default()));

    let mut tasks = HashMap::new();
    for (name, task_val) in raw.tasks {
        let merged = merge_yaml(&defaults_val, &task_val);
        let resolved = resolve_env_vars(&merged);

        let mut task: TaskConfig = serde_yaml::from_value(resolved)
            .with_context(|| format!("Failed to parse task '{name}'"))?;

        task.name = name.clone();

        if let Some(no_verify) = task.no_verify_ssl {
            task.verify_ssl = !no_verify;
        }

        // Validate auth reference exists
        if let Some(ref auth_name) = task.auth {
            if !auth.contains_key(auth_name) {
                return Err(anyhow!(
                    "Task '{name}' references unknown auth provider '{auth_name}'"
                ));
            }
        }

        tasks.insert(name, task);
    }

    Ok(Config { auth, tasks })
}

pub fn find_config() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("yabench.yaml"),
        PathBuf::from("yabench.yml"),
    ];

    candidates.iter().find(|p| p.exists()).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_size_plain() {
        assert_eq!(parse_size("0").unwrap(), 0);
        assert_eq!(parse_size("100").unwrap(), 100);
        assert_eq!(parse_size("1000000").unwrap(), 1_000_000);
    }

    #[test]
    fn parse_size_suffixes() {
        assert_eq!(parse_size("4K").unwrap(), 4_000);
        assert_eq!(parse_size("4k").unwrap(), 4_000);
        assert_eq!(parse_size("128K").unwrap(), 128_000);
        assert_eq!(parse_size("1M").unwrap(), 1_000_000);
        assert_eq!(parse_size("1m").unwrap(), 1_000_000);
    }

    #[test]
    fn parse_size_fractional() {
        assert_eq!(parse_size("1.5K").unwrap(), 1_500);
        assert_eq!(parse_size("0.5M").unwrap(), 500_000);
    }

    #[test]
    fn parse_size_whitespace() {
        assert_eq!(parse_size(" 4K ").unwrap(), 4_000);
    }

    #[test]
    fn parse_size_errors() {
        assert!(parse_size("").is_err());
        assert!(parse_size("abc").is_err());
        assert!(parse_size("-1").is_err());
        assert!(parse_size("4G").is_err());
    }

    #[test]
    fn parse_size_u32_overflow() {
        assert!(parse_size_u32("5G").is_err()); // unknown suffix anyway
        assert!(parse_size_u32("10M").is_ok());
        // 5e9 exceeds u32::MAX (4.29e9)
        assert!(parse_size_u32("5000M").is_err());
    }
}
