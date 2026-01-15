"""OpenAI-compatible streaming client with custom auth."""

import time
import json
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal


@dataclass
class ParsedChunk:
    """Result of parsing a streaming chunk."""
    content: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    is_done: bool = False


class StreamParser(ABC):
    """Base class for streaming response parsers."""

    @abstractmethod
    def parse_line(self, line: str) -> ParsedChunk | None:
        """Parse a single line from the stream. Returns None to skip the line."""
        pass


class OpenAIStreamParser(StreamParser):
    """Parser for OpenAI-compatible streaming format.

    Format: data: {"choices":[{"delta":{"content":"..."}}]}
    """

    def parse_line(self, line: str) -> ParsedChunk | None:
        if not line.startswith("data:"):
            return None

        data = line[5:].lstrip()
        if data == "[DONE]":
            return ParsedChunk(is_done=True)

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return None

        result = ParsedChunk()

        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            result.content = delta.get("content")

        usage = chunk.get("usage")
        if usage:
            result.input_tokens = usage.get("prompt_tokens")
            result.output_tokens = usage.get("completion_tokens")

        return result


class MasStreamParser(StreamParser):
    """Parser for MAS streaming format.

    Format: data:{"choices":[{"message":{"content":"..."}}]}
    Events: event:{"usage":{"completionTokens":...,"promptTokens":...}}
    """

    def parse_line(self, line: str) -> ParsedChunk | None:
        # Handle usage events
        if line.startswith("event:"):
            try:
                event_data = json.loads(line[6:])
                usage = event_data.get("usage", {})
                if usage:
                    return ParsedChunk(
                        input_tokens=usage.get("promptTokens"),
                        output_tokens=usage.get("completionTokens"),
                    )
            except json.JSONDecodeError:
                pass
            return None

        if not line.startswith("data:"):
            return None

        data = line[5:].lstrip()
        if data == "[DONE]":
            return ParsedChunk(is_done=True)

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return None

        result = ParsedChunk()

        choices = chunk.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            result.content = message.get("content")

        return result


def detect_parser(line: str) -> StreamParser:
    """Detect which parser to use based on the first data line."""
    if line.startswith("data:"):
        data = line[5:].lstrip()
        try:
            chunk = json.loads(data)
            choices = chunk.get("choices", [])
            if choices:
                if "delta" in choices[0]:
                    return OpenAIStreamParser()
                elif "message" in choices[0]:
                    return MasStreamParser()
        except json.JSONDecodeError:
            pass
    return OpenAIStreamParser()  # Default to OpenAI format


ParserType = Literal["openai", "mas", "auto"]


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
    def prefill_tps(self) -> float:
        """Prefill tokens per second (input_tokens / TTFT)."""
        if self.ttft is None or self.ttft <= 0 or self.input_tokens <= 0:
            return 0.0
        return self.input_tokens / self.ttft

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
        parser: ParserType = "auto",
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.parser_type = parser

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

                    # Initialize parser
                    parser: StreamParser | None = None
                    if self.parser_type == "openai":
                        parser = OpenAIStreamParser()
                    elif self.parser_type == "mas":
                        parser = MasStreamParser()
                    # else: auto-detect from first line

                    async for line in response.aiter_lines():
                        # Auto-detect parser from first data line
                        if parser is None:
                            parser = detect_parser(line)

                        parsed = parser.parse_line(line)
                        if parsed is None:
                            continue

                        if parsed.is_done:
                            break

                        # Update usage metrics if provided
                        if parsed.input_tokens is not None:
                            metrics.input_tokens = parsed.input_tokens
                        if parsed.output_tokens is not None:
                            metrics.output_tokens = parsed.output_tokens

                        # Handle content
                        if parsed.content:
                            now = time.perf_counter()

                            if not first_token_received:
                                metrics.ttft = now - start_time
                                first_token_received = True
                            else:
                                metrics.inter_token_latencies.append(now - last_token_time)

                            last_token_time = now
                            metrics.output_tokens += 1
                            yield parsed.content, metrics

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
