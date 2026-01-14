"""Dataset loading for realistic benchmarks."""

import json
import random
from pathlib import Path
from dataclasses import dataclass

import httpx


@dataclass
class Prompt:
    """A prompt with optional conversation context."""
    text: str
    system: str | None = None


# Local datasets bundled with the package (for offline use)
LOCAL_DATASETS = {
    "sample-en": "sample-en.jsonl",      # 50 English prompts (basic)
    "sample-zh": "sample-zh.jsonl",      # 50 Chinese prompts (basic)
    "alpaca-en": "alpaca-en-500.jsonl",  # 500 English prompts from Stanford Alpaca
    "alpaca-zh": "alpaca-zh-500.jsonl",  # 500 Chinese prompts from Alpaca-GPT4-Chinese
}

# Remote datasets (downloaded on demand from HuggingFace)
REMOTE_DATASETS = {
    # English - large conversation datasets
    "sharegpt": "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
    "sharegpt-small": "https://huggingface.co/datasets/shibing624/sharegpt_gpt4/resolve/main/sharegpt_gpt4.jsonl",
    # Chinese - large instruction datasets
    "belle": "https://huggingface.co/datasets/BelleGroup/train_1M_CN/resolve/main/Belle_open_source_1M.json",
    "firefly": "https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M/resolve/main/firefly-train-1.1M.jsonl",
}

# All built-in datasets
BUILTIN_DATASETS = {**LOCAL_DATASETS, **REMOTE_DATASETS}


def get_local_datasets_dir() -> Path:
    """Get the directory containing bundled datasets."""
    # Look relative to this file's location
    return Path(__file__).parent.parent / "datasets"


def load_json_dataset(path: Path | str, max_prompts: int | None = None) -> list[Prompt]:
    """
    Load JSON dataset. Supports formats:
    - ShareGPT: [{"conversations": [{"from": "human", "value": "..."}]}]
    - Alpaca: [{"instruction": "...", "input": "...", "output": "..."}]
    - BELLE: [{"instruction": "...", "output": "..."}]
    """
    with open(path) as f:
        data = json.load(f)

    prompts = []
    for item in data:
        text = None

        # ShareGPT format
        if "conversations" in item:
            for msg in item["conversations"]:
                if msg.get("from") == "human":
                    text = msg.get("value", "").strip()
                    break
        # Alpaca/BELLE format
        elif "instruction" in item:
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            if input_text:
                text = f"{instruction}\n\n{input_text}"
            else:
                text = instruction
        # Simple prompt format
        elif "prompt" in item:
            text = item.get("prompt", "").strip()
        elif "text" in item:
            text = item.get("text", "").strip()

        if text and len(text) > 10:
            prompts.append(Prompt(text=text))

        if max_prompts and len(prompts) >= max_prompts:
            break

    return prompts


def load_jsonl(path: Path | str, max_prompts: int | None = None) -> list[Prompt]:
    """
    Load JSONL dataset. Supports formats:
    - {"prompt": "..."} or {"text": "..."} or {"content": "..."}
    - {"instruction": "...", "input": "..."} (Alpaca/BELLE/Firefly)
    - {"messages": [{"role": "user", "content": "..."}]}
    - ShareGPT JSONL: {"conversations": [{"from": "human", "value": "..."}]}
    """
    prompts = []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = None
            system = None

            # Try different formats
            if "prompt" in item:
                text = item["prompt"]
            elif "instruction" in item:
                # Alpaca/BELLE/Firefly format
                instruction = item.get("instruction", "").strip()
                input_text = item.get("input", "").strip()
                if input_text:
                    text = f"{instruction}\n\n{input_text}"
                else:
                    text = instruction
            elif "text" in item:
                text = item["text"]
            elif "content" in item:
                text = item["content"]
            elif "question" in item:
                text = item["question"]
            elif "messages" in item:
                # OpenAI messages format
                for msg in item["messages"]:
                    if msg.get("role") == "system":
                        system = msg.get("content")
                    elif msg.get("role") == "user":
                        text = msg.get("content")
                        break
            elif "conversations" in item:
                # ShareGPT format
                for msg in item["conversations"]:
                    if msg.get("from") == "human":
                        text = msg.get("value")
                        break

            if text and len(text.strip()) > 10:
                prompts.append(Prompt(text=text.strip(), system=system))

            if max_prompts and len(prompts) >= max_prompts:
                break

    return prompts


def load_txt(path: Path | str, max_prompts: int | None = None) -> list[Prompt]:
    """Load plain text file, one prompt per line."""
    prompts = []

    with open(path) as f:
        for line in f:
            text = line.strip()
            if text and len(text) > 10:
                prompts.append(Prompt(text=text))

            if max_prompts and len(prompts) >= max_prompts:
                break

    return prompts


async def download_dataset(url: str, dest: Path) -> None:
    """Download a dataset file."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded datasets."""
    cache = Path.home() / ".cache" / "yabench" / "datasets"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def get_dataset_path(name: str) -> Path:
    """Get the local path for a built-in dataset."""
    # Check if it's a local bundled dataset
    if name in LOCAL_DATASETS:
        return get_local_datasets_dir() / LOCAL_DATASETS[name]

    # Remote dataset - return cache path
    url = REMOTE_DATASETS[name]
    ext = ".jsonl" if url.endswith(".jsonl") else ".json"
    return get_cache_dir() / f"{name}{ext}"


async def download_builtin(name: str, quiet: bool = False) -> Path:
    """Download a built-in dataset (skips local datasets)."""
    if name not in BUILTIN_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {', '.join(BUILTIN_DATASETS.keys())}")

    # Local datasets don't need download
    if name in LOCAL_DATASETS:
        path = get_dataset_path(name)
        if not quiet:
            print(f"  {name}: bundled locally ({path})")
        return path

    path = get_dataset_path(name)
    if path.exists():
        if not quiet:
            print(f"  {name}: already downloaded ({path})")
        return path

    if not quiet:
        print(f"  {name}: downloading...")

    await download_dataset(REMOTE_DATASETS[name], path)

    if not quiet:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {name}: done ({size_mb:.1f} MB)")

    return path


async def download_all_datasets(quiet: bool = False) -> dict[str, Path]:
    """Download all built-in datasets."""
    results = {}
    for name in BUILTIN_DATASETS:
        try:
            results[name] = await download_builtin(name, quiet)
        except Exception as e:
            if not quiet:
                print(f"  {name}: failed - {e}")
    return results


async def load_dataset(
    source: str,
    num_prompts: int = 100,
    shuffle: bool = True,
    seed: int = 42,
) -> list[Prompt]:
    """
    Load prompts from a dataset.

    Args:
        source: Dataset name (e.g., "sharegpt") or path to file
        num_prompts: Number of prompts to load
        shuffle: Whether to shuffle prompts
        seed: Random seed for reproducible shuffling

    Returns:
        List of Prompt objects
    """
    path = Path(source)

    # Check if it's a built-in dataset name
    if source in BUILTIN_DATASETS:
        path = get_dataset_path(source)

        if not path.exists():
            if source in LOCAL_DATASETS:
                raise FileNotFoundError(f"Local dataset not found: {path}")
            print(f"Downloading {source} dataset...")
            await download_builtin(source, quiet=True)
            print(f"Saved to {path}")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")

    # Load based on file extension
    suffix = path.suffix.lower()
    # Load more than needed to allow filtering/shuffling
    max_load = num_prompts * 3 if shuffle else num_prompts

    if suffix == ".json":
        prompts = load_json_dataset(path, max_load)
    elif suffix == ".jsonl":
        prompts = load_jsonl(path, max_load)
    elif suffix == ".txt":
        prompts = load_txt(path, max_load)
    else:
        # Try JSONL first, then plain text
        try:
            prompts = load_jsonl(path, max_load)
        except Exception:
            prompts = load_txt(path, max_load)

    if not prompts:
        raise ValueError(f"No valid prompts found in {source}")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(prompts)

    return prompts[:num_prompts]


def prompts_to_strings(prompts: list[Prompt]) -> list[str]:
    """Convert Prompt objects to plain strings."""
    return [p.text for p in prompts]
