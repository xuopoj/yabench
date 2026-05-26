# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

yabench is a Rust CLI that benchmarks OpenAI-compatible LLM APIs. Single static binary, no runtime deps. Measures TTFT / ITL / E2E latency and prefill / output throughput on streaming chat completions. Released as cross-platform binaries via GitHub Actions (musl static on Linux, native on macOS/Windows; rustls — no OpenSSL).

`README.md` is in Chinese and user-facing. This file is the developer/Claude reference.

## Commands

```bash
cargo build --release       # build the binary at target/release/yabench
cargo build                 # debug build (faster compile, slower runtime)
cargo test                  # run unit tests
cargo test parse_size       # run a specific test by name
```

Unit tests live only in `src/config.rs` and cover the size-suffix parser (`4K`, `1.5M`, etc.). Network paths are smoke-tested manually against a real endpoint — there is no integration test harness.

Common runtime invocations once built:

```bash
./target/release/yabench --init                            # write yabench.yaml from template
./target/release/yabench <task>                            # run a configured task
./target/release/yabench --base-url URL -n N -c C          # direct mode (no config)
./target/release/yabench <task> --chat "hi"                # one-shot streaming sanity check
./target/release/yabench <task> --perf-report              # concurrency sweep + markdown report
./target/release/yabench <task> --prefix-tokens 2000       # exercise prefix caching
./target/release/yabench <task> --multi-turn --dataset sharegpt-small   # realistic chat replay
./target/release/yabench <task> --perf-matrix              # input×output×concurrency matrix
./target/release/yabench <task> --perf-matrix --matrix-n 20  # 20 requests per cell
```

## Architecture

### Single-pass data flow

```
CLI args ──┐
           ├─► get_task_config → TaskConfig
yabench.yaml ──┘                  │
                                  ▼
                  auth resolution (api_key | token | iam.rs)
                                  │
                                  ▼
                              OpenAIClient
                              ├─ system_prefix (optional)
                              └─ chars_per_token (auto-calibrated)
                                  │
                                  ▼
              prompts → Vec<serde_json::Value> (chat-messages arrays)
                  · benchmark::generate_prompts          (synthetic)
                  · datasets::load_dataset               (single-turn)
                  · datasets::load_multi_turn            (growing-prefix replay)
                                  │
                                  ▼
                       benchmark::run_benchmark
                       ├─ tokio::Semaphore for concurrency
                       ├─ warmup phase
                       └─ retry with exponential backoff
                                  │
                                  ▼
                       BenchmarkResult + per-request metrics
                                  │
                                  ▼
            print summary │ write JSON │ append CSV │ markdown report (perf.rs)
```

### Module map

| Module | Role |
|---|---|
| `main.rs` | CLI parsing, dispatch (chat / init / list / download / print-dataset / perf-report / run-task). The big switch lives here. |
| `client.rs` | One `OpenAIClient` struct. Streams `/chat/completions`. Auto-detects two SSE formats: `OpenAI` (delta-style) and `MAS` (Huawei MindIE message-style). Owns `system_prefix` and `chars_per_token`. `calibrate_tokenizer` probes the server with one short request to learn its chars/token ratio. |
| `benchmark.rs` | `run_benchmark` is the inner loop — spawns one tokio task per request gated by a semaphore. Computes percentile stats. |
| `datasets.rs` | Embedded datasets (`include_bytes!`) + remote (HF Hub) + local-file loaders. `parse_item` understands many JSONL schemas. `load_multi_turn` expands ShareGPT/OpenAI-format conversations into round-robin growing-prefix request lists. |
| `config.rs` | YAML config with `defaults` / `auth` / `tasks` sections. `${ENV_VAR}` interpolation; per-task merge of defaults. `parse_size` / `parse_size_u32` accept `4K`/`1M` suffixes — used by both serde and clap. |
| `iam.rs` | One function: fetches a Huawei Cloud `X-Subject-Token` via `/v3/auth/tokens`. |
| `perf.rs` | `--perf-report` suite: c=1,2,4,8 sweep; between levels runs an **eviction storm** then a cool-down. Detects the saturation knee. Emits Markdown. Also `--perf-matrix`: 3D sweep over input_tokens × output_tokens × concurrency (default grid: input [1K,4K,16K,64K,128K] × output [256,1K,4K,16K,64K] × c [1,4,8] = 75 cells). |

### Key cross-file invariants

- **Everything routes through `messages: Vec<serde_json::Value>`.** Each element is a chat-completions `messages` array. Synthetic-prompt callers wrap via `benchmark::prompts_to_messages`; multi-turn loaders construct them directly. The client never sees raw prompt strings (except via the convenience wrapper `chat()`).

- **`stream_options.include_usage: true` is always sent.** Most servers (vLLM, SGLang, OpenAI, MindIE) honor it and emit a final usage block with both `prompt_tokens` and `completion_tokens`. The chars/ratio fallback only fires when the server omits usage; per-request `input_tokens_estimated` / `output_tokens_estimated` flags track which numbers were estimated and surface as `(N estimated)` in the summary.

- **`chars_per_token` is calibrated once at startup** via `OpenAIClient::calibrate_tokenizer` — a single non-streaming POST with `max_tokens=1` reads `usage.prompt_tokens`. The result feeds both synthetic prompt sizing and the fallback estimator. Calibration is best-effort; falls back to 4.0 if the probe fails.

- **`system_prefix` is set on the client, not per-request.** `stream_complete_with` prepends it as a `{"role":"system", …}` message inside the function. Eviction-storm requests intentionally use the same client and inherit the prefix — they target the *user-side* portion of the cache, not the system message.

- **Auth flow:** CLI flags override task config; task config references a named provider in the `auth:` map; for `iam` providers the token is fetched once before the run. `task.token` and `task.api_key` are populated before the client is built.

### Why the perf suite is non-trivial

The sweep was producing inconsistent TTFT curves because vLLM's prefix cache (LRU-on-capacity, not time-based) leaked warm entries between levels — a c=4 run would prime the cache for c=8, making c=8 look artificially fast. Three mechanisms in `perf.rs` keep level boundaries clean:

1. **Round-robin multi-turn replay** in `datasets::load_multi_turn`: requests rotate across conversations (`conv1.t1, conv2.t1, …, conv1.t2, conv2.t2, …`) so concurrent requests within a single level don't all share one growing prefix. Default cap of 5 user-turns per conversation forces diversity.

2. **Eviction storm** between levels (`eviction_storm` in `perf.rs`): N parallel requests with unique nonce-padded prompts (`evict-<seed1>-<seed2> ` repeated to target length) push prior-level cache entries out via LRU. Default 8 × ~4000 tokens, `max_tokens=4` so decode is trivial.

3. **Cool-down sleep** (default 5s) after the storm lets in-flight requests drain before the next level starts.

Set any of `PerfSuiteConfig::cooldown_secs`, `eviction_storm_requests`, or `eviction_storm_tokens` to 0 to disable that mechanism.

## Config and secrets

`yabench.yaml` is **gitignored** because it commonly contains API keys / IAM passwords. The committed template is `yabench.example.yaml`; `--init` copies it via `include_str!`. Tasks reference auth providers by name; providers support `${ENV_VAR}` interpolation for credentials.

## Release pipeline

`.github/workflows/release.yml` triggers on `v*` tags. Builds 5 targets: linux x86_64/arm64 (both musl, fully static), darwin x86_64/arm64, windows x86_64. Static linking is why `reqwest` uses `default-features = false` + `rustls-tls` — removes the OpenSSL system dependency.
