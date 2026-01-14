"""Config file handling for yabench."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskConfig:
    """Configuration for a benchmark task."""
    name: str
    base_url: str
    model: str | None = None
    token: str | None = None
    api_key: str | None = None
    num_requests: int = 10
    concurrency: int = 1
    max_tokens: int = 256
    input_tokens: int = 100
    timeout: float = 120.0
    verify_ssl: bool = True
    dataset: str | None = None  # Dataset name or path
    shuffle: bool = True  # Shuffle dataset prompts
    seed: int = 42  # Random seed for reproducibility


@dataclass
class Config:
    """Root configuration."""
    tasks: dict[str, TaskConfig] = field(default_factory=dict)

    # Global defaults that tasks can inherit
    defaults: dict[str, Any] = field(default_factory=dict)


def _resolve_env_vars(value: Any) -> Any:
    """Resolve ${ENV_VAR} references in string values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, "")
    return value


def load_config(path: Path) -> Config:
    """Load config from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    defaults = data.get("defaults", {})
    tasks_data = data.get("tasks", {})

    tasks = {}
    for name, task_data in tasks_data.items():
        # Merge defaults with task-specific config
        merged = {**defaults, **task_data}

        # Resolve environment variables
        for key, value in merged.items():
            merged[key] = _resolve_env_vars(value)

        # Handle verify_ssl / no_verify_ssl
        if "no_verify_ssl" in merged:
            merged["verify_ssl"] = not merged.pop("no_verify_ssl")

        tasks[name] = TaskConfig(name=name, **merged)

    return Config(tasks=tasks, defaults=defaults)


def find_config() -> Path | None:
    """Find config file in current directory or home."""
    candidates = [
        Path("yabench.yaml"),
        Path("yabench.yml"),
        Path.home() / ".yabench.yaml",
        Path.home() / ".yabench.yml",
        Path.home() / ".config" / "yabench" / "config.yaml",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None
