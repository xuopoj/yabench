"""Benchmark runner with concurrent requests."""

import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Callable

from .client import OpenAIClient, RequestMetrics


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    num_requests: int = 0
    num_completed: int = 0
    num_errors: int = 0
    total_duration: float = 0.0

    # Latency stats (seconds)
    ttft_mean: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0

    e2e_latency_mean: float = 0.0
    e2e_latency_p50: float = 0.0
    e2e_latency_p95: float = 0.0
    e2e_latency_p99: float = 0.0

    inter_token_latency_mean: float = 0.0

    # Throughput
    tokens_per_second: float = 0.0  # Total output tokens / total duration
    requests_per_second: float = 0.0

    # Token counts
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Individual request metrics
    request_metrics: list[RequestMetrics] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Benchmark Results",
            f"=" * 40,
            f"Requests: {self.num_completed}/{self.num_requests} completed ({self.num_errors} errors)",
            f"Duration: {self.total_duration:.2f}s",
            f"",
            f"Throughput:",
            f"  Requests/sec: {self.requests_per_second:.2f}",
            f"  Tokens/sec:   {self.tokens_per_second:.2f}",
            f"",
            f"Latency (seconds):",
            f"  TTFT:     mean={self.ttft_mean:.3f} p50={self.ttft_p50:.3f} p95={self.ttft_p95:.3f} p99={self.ttft_p99:.3f}",
            f"  E2E:      mean={self.e2e_latency_mean:.3f} p50={self.e2e_latency_p50:.3f} p95={self.e2e_latency_p95:.3f} p99={self.e2e_latency_p99:.3f}",
            f"  ITL:      mean={self.inter_token_latency_mean:.4f}",
            f"",
            f"Tokens:",
            f"  Input:  {self.total_input_tokens}",
            f"  Output: {self.total_output_tokens}",
        ]
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for err in self.errors[:5]:
                lines.append(f"  - {err[:80]}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors) - 5} more")
        return "\n".join(lines)


def _percentile(data: list[float], p: float) -> float:
    """Calculate percentile of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def _compute_stats(result: BenchmarkResult) -> None:
    """Compute aggregate statistics from individual request metrics."""
    completed = [m for m in result.request_metrics if m.error is None]

    if not completed:
        return

    # TTFT stats
    ttfts = [m.ttft for m in completed if m.ttft is not None]
    if ttfts:
        result.ttft_mean = statistics.mean(ttfts)
        result.ttft_p50 = _percentile(ttfts, 50)
        result.ttft_p95 = _percentile(ttfts, 95)
        result.ttft_p99 = _percentile(ttfts, 99)

    # E2E latency stats
    latencies = [m.total_time for m in completed]
    if latencies:
        result.e2e_latency_mean = statistics.mean(latencies)
        result.e2e_latency_p50 = _percentile(latencies, 50)
        result.e2e_latency_p95 = _percentile(latencies, 95)
        result.e2e_latency_p99 = _percentile(latencies, 99)

    # Inter-token latency
    all_itls = []
    for m in completed:
        all_itls.extend(m.inter_token_latencies)
    if all_itls:
        result.inter_token_latency_mean = statistics.mean(all_itls)

    # Token counts
    result.total_input_tokens = sum(m.input_tokens for m in completed)
    result.total_output_tokens = sum(m.output_tokens for m in completed)

    # Throughput
    if result.total_duration > 0:
        result.tokens_per_second = result.total_output_tokens / result.total_duration
        result.requests_per_second = result.num_completed / result.total_duration


async def run_benchmark(
    client: OpenAIClient,
    prompts: list[str],
    model: str | None = None,
    max_tokens: int = 256,
    concurrency: int = 1,
    on_request_complete: Callable[[int, RequestMetrics], None] | None = None,
) -> BenchmarkResult:
    """
    Run benchmark with the given prompts.

    Args:
        client: OpenAI-compatible client
        prompts: List of prompts to send
        model: Model name (optional, some APIs have defaults)
        max_tokens: Max tokens per request
        concurrency: Number of concurrent requests
        on_request_complete: Optional callback for progress updates

    Returns:
        BenchmarkResult with aggregated metrics
    """
    result = BenchmarkResult(num_requests=len(prompts))
    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0

    async def run_single(idx: int, prompt: str) -> RequestMetrics:
        nonlocal completed_count
        async with semaphore:
            _, metrics = await client.complete(prompt, model, max_tokens)
            completed_count += 1
            if on_request_complete:
                on_request_complete(completed_count, metrics)
            return metrics

    import time
    start = time.perf_counter()

    tasks = [run_single(i, p) for i, p in enumerate(prompts)]
    metrics_list = await asyncio.gather(*tasks)

    result.total_duration = time.perf_counter() - start
    result.request_metrics = list(metrics_list)

    # Count completed vs errors
    for m in metrics_list:
        if m.error:
            result.num_errors += 1
            result.errors.append(m.error)
        else:
            result.num_completed += 1

    _compute_stats(result)
    return result


def generate_prompts(
    num_prompts: int,
    input_tokens: int = 100,
) -> list[str]:
    """Generate prompts of approximately the given token count."""
    # Simple prompt generation - repeat a phrase to approximate token count
    # Roughly 4 chars per token
    base = "Write a detailed explanation about the following topic. Be thorough and comprehensive. "
    filler = "Please provide more details. "

    target_chars = input_tokens * 4
    prompt = base
    while len(prompt) < target_chars:
        prompt += filler

    return [prompt[:target_chars] for _ in range(num_prompts)]
