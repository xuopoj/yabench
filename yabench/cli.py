"""CLI for yabench."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .client import OpenAIClient, RequestMetrics
from .benchmark import run_benchmark, generate_prompts, BenchmarkResult
from .config import load_config, find_config, TaskConfig
from .datasets import load_dataset, prompts_to_strings, download_all_datasets, download_builtin, BUILTIN_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="yabench - Benchmark OpenAI-compatible LLM APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets for offline use
  yabench --download

  # Run a task from config file
  yabench my-task

  # Run with dataset
  yabench --base-url http://localhost:8000/v1 --dataset sharegpt -n 100

  # List available tasks
  yabench --list
        """,
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="Task name from config file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Config file path (default: yabench.yaml)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tasks from config",
    )
    parser.add_argument(
        "--download",
        nargs="?",
        const="all",
        metavar="DATASET",
        help="Download datasets. Use --download for all, or --download NAME for specific",
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for the OpenAI-compatible API",
    )
    parser.add_argument(
        "--model",
        help="Model name (optional if API has default)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("YABENCH_TOKEN"),
        help="Auth token (sent as X-Auth-Token header). Env: YABENCH_TOKEN",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (sent as Bearer token). Env: OPENAI_API_KEY",
    )
    parser.add_argument(
        "-n", "--num-requests",
        type=int,
        help="Number of requests to send (default: 10)",
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        help="Number of concurrent requests (default: 1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Max tokens per response (default: 256)",
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        help="Approximate input tokens per prompt (default: 100, ignored if --dataset used)",
    )
    parser.add_argument(
        "--dataset",
        help=f"Dataset name or file path. Built-in: {', '.join(BUILTIN_DATASETS.keys())}",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle dataset prompts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible prompt selection (default: 42)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for JSON results",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "-k", "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification",
    )

    return parser.parse_args()


def result_to_dict(result: BenchmarkResult) -> dict:
    """Convert BenchmarkResult to JSON-serializable dict."""
    return {
        "num_requests": result.num_requests,
        "num_completed": result.num_completed,
        "num_errors": result.num_errors,
        "total_duration": result.total_duration,
        "throughput": {
            "requests_per_second": result.requests_per_second,
            "tokens_per_second": result.tokens_per_second,
        },
        "latency": {
            "ttft": {
                "mean": result.ttft_mean,
                "p50": result.ttft_p50,
                "p95": result.ttft_p95,
                "p99": result.ttft_p99,
            },
            "e2e": {
                "mean": result.e2e_latency_mean,
                "p50": result.e2e_latency_p50,
                "p95": result.e2e_latency_p95,
                "p99": result.e2e_latency_p99,
            },
            "inter_token_mean": result.inter_token_latency_mean,
        },
        "tokens": {
            "input": result.total_input_tokens,
            "output": result.total_output_tokens,
        },
        "errors": result.errors,
    }


def get_task_config(args: argparse.Namespace) -> TaskConfig | None:
    """Get task config from args or config file."""
    # Try to load config file
    config_path = args.config or find_config()
    config = load_config(config_path) if config_path else None

    # If task name provided, load from config
    if args.task:
        if not config:
            print(f"Error: No config file found", file=sys.stderr)
            return None
        if args.task not in config.tasks:
            print(f"Error: Task '{args.task}' not found in config", file=sys.stderr)
            print(f"Available tasks: {', '.join(config.tasks.keys())}", file=sys.stderr)
            return None

        task = config.tasks[args.task]

        # CLI args override config
        if args.base_url:
            task.base_url = args.base_url
        if args.model:
            task.model = args.model
        if args.token:
            task.token = args.token
        if args.api_key:
            task.api_key = args.api_key
        if args.num_requests:
            task.num_requests = args.num_requests
        if args.concurrency:
            task.concurrency = args.concurrency
        if args.max_tokens:
            task.max_tokens = args.max_tokens
        if args.input_tokens:
            task.input_tokens = args.input_tokens
        if args.timeout:
            task.timeout = args.timeout
        if args.no_verify_ssl:
            task.verify_ssl = False
        if args.dataset:
            task.dataset = args.dataset
        if args.no_shuffle:
            task.shuffle = False
        if args.seed is not None:
            task.seed = args.seed

        return task

    # No task name - use CLI args directly
    if not args.base_url:
        print("Error: --base-url required (or use a task name from config)", file=sys.stderr)
        return None

    return TaskConfig(
        name="cli",
        base_url=args.base_url,
        model=args.model,
        token=args.token,
        api_key=args.api_key,
        num_requests=args.num_requests or 10,
        concurrency=args.concurrency or 1,
        max_tokens=args.max_tokens or 256,
        input_tokens=args.input_tokens or 100,
        timeout=args.timeout or 120.0,
        verify_ssl=not args.no_verify_ssl,
        dataset=args.dataset,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )


async def run_task(task: TaskConfig, output: str | None, quiet: bool) -> int:
    """Run a benchmark task."""
    client = OpenAIClient(
        base_url=task.base_url,
        token=task.token,
        api_key=task.api_key,
        timeout=task.timeout,
        verify_ssl=task.verify_ssl,
    )

    if not quiet:
        print(f"yabench - Running task: {task.name}")
        print(f"  URL: {task.base_url}")
        print(f"  Model: {task.model or '(default)'}")
        print(f"  Requests: {task.num_requests} (concurrency: {task.concurrency})")
        print(f"  Max tokens: {task.max_tokens}")
        if task.dataset:
            print(f"  Dataset: {task.dataset}")
        print()

    # Load prompts from dataset or generate synthetic ones
    if task.dataset:
        if not quiet:
            print("Loading dataset...")
        dataset_prompts = await load_dataset(
            task.dataset, task.num_requests, task.shuffle, task.seed
        )
        prompts = prompts_to_strings(dataset_prompts)
        if not quiet:
            print(f"Loaded {len(prompts)} prompts (seed={task.seed})")
            print()
    else:
        prompts = generate_prompts(task.num_requests, task.input_tokens)

    def on_complete(completed: int, metrics: RequestMetrics) -> None:
        if quiet:
            return
        status = "OK" if metrics.error is None else f"ERR: {metrics.error[:30]}"
        print(f"  [{completed}/{task.num_requests}] {status}", end="\r")

    result = await run_benchmark(
        client=client,
        prompts=prompts,
        model=task.model,
        max_tokens=task.max_tokens,
        concurrency=task.concurrency,
        on_request_complete=on_complete,
    )

    if not quiet:
        print()
        print()
        print(result.summary())

    if output:
        output_path = Path(output)
        output_path.write_text(json.dumps(result_to_dict(result), indent=2))
        if not quiet:
            print(f"\nResults saved to {output}")

    return 0 if result.num_errors == 0 else 1


async def main_async() -> int:
    args = parse_args()

    # Handle --list
    if args.list:
        config_path = args.config or find_config()
        if not config_path:
            print("No config file found", file=sys.stderr)
            return 1
        config = load_config(config_path)
        print(f"Tasks in {config_path}:")
        for name, task in config.tasks.items():
            print(f"  {name}: {task.base_url}")
        return 0

    # Handle --download
    if args.download:
        print("Downloading datasets...")
        if args.download == "all":
            await download_all_datasets()
        else:
            try:
                await download_builtin(args.download)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        print("\nDone. Datasets cached in ~/.cache/yabench/datasets/")
        return 0

    task = get_task_config(args)
    if not task:
        return 1

    return await run_task(task, args.output, args.quiet)


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
