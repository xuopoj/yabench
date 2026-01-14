"""OpenAI-compatible streaming client with custom auth."""

import time
import json
import httpx
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class RequestMetrics:
    """Metrics collected from a single request."""
    ttft: float | None = None  # Time to first token (seconds)
    total_time: float = 0.0  # End-to-end latency (seconds)
    input_tokens: int = 0
    output_tokens: int = 0
    inter_token_latencies: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def tokens_per_second(self) -> float:
        """Output tokens per second."""
        if self.total_time <= 0 or self.output_tokens <= 0:
            return 0.0
        return self.output_tokens / self.total_time

    @property
    def mean_inter_token_latency(self) -> float:
        """Mean time between tokens."""
        if not self.inter_token_latencies:
            return 0.0
        return sum(self.inter_token_latencies) / len(self.inter_token_latencies)


class OpenAIClient:
    """Async client for OpenAI-compatible APIs with custom auth."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Auth-Token"] = self.token
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def stream_chat(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> AsyncIterator[tuple[str, RequestMetrics]]:
        """Stream chat completion and yield (content_chunk, metrics)."""
        metrics = RequestMetrics()
        start_time = time.perf_counter()
        last_token_time = start_time
        first_token_received = False

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if model:
            payload["model"] = model

        try:
            async with httpx.AsyncClient(timeout=self.timeout, verify=self.verify_ssl) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        text = await response.aread()
                        metrics.error = f"HTTP {response.status_code}: {text.decode()}"
                        metrics.total_time = time.perf_counter() - start_time
                        yield "", metrics
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            try:
                                usage = json.loads(line[6:])
                                if usage:
                                    metrics.input_tokens = usage.get("promptTokens", 0)
                                    if "completionTokens" in usage:
                                        metrics.output_tokens = usage["completionTokens"]
                            except json.JSONDecodeError:
                                continue
                        if not line.startswith("data:"):
                            continue

                        data = line[5:]  # Strip "data: " prefix
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        # Extract content from delta
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        if content:
                            now = time.perf_counter()

                            if not first_token_received:
                                metrics.ttft = now - start_time
                                first_token_received = True
                            else:
                                metrics.inter_token_latencies.append(now - last_token_time)

                            last_token_time = now
                            metrics.output_tokens += 1
                            yield content, metrics

                        # Check for usage info (some APIs include it)
                        usage = chunk.get("usage")
                        if usage:
                            metrics.input_tokens = usage.get("promptTokens", 0)
                            if "completionTokens" in usage:
                                metrics.output_tokens = usage["completionTokens"]

        except httpx.TimeoutException:
            metrics.error = "Request timed out"
        except httpx.RequestError as e:
            metrics.error = f"Request error: {e}"
        except Exception as e:
            metrics.error = f"Unexpected error: {e}"

        metrics.total_time = time.perf_counter() - start_time
        yield "", metrics

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[str, RequestMetrics]:
        """Complete a chat request and return (full_response, metrics)."""
        full_response = []
        metrics = RequestMetrics()

        async for chunk, metrics in self.stream_chat(prompt, model, max_tokens, temperature):
            full_response.append(chunk)

        return "".join(full_response), metrics
