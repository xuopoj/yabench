use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
    pub max_tokens: u32,
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
            input_tokens: 100,
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
