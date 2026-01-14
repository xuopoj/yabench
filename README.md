# Yet Another LLM Bencmark Tool

## Installation

```bash
uv venv
uv pip install -e
```

## Usage

```bash
# create a task in yabench.yaml, and then run task:
yabench <task_name>

# or run with command
yabench --base-url https://172.25.61.164:31004/v1/cbfda6be7955455e9c49b45ee556f427/deployments/ae146460-d669-42db-ab4a-f244e464b841 -k -n 100 -c 10

```

## Help

```bash
yabench --help
```

```
usage: yabench [-h] [--config CONFIG] [--list] [--download [DATASET]] [--base-url BASE_URL] [--model MODEL] [--token TOKEN] [--api-key API_KEY]
               [-n NUM_REQUESTS] [-c CONCURRENCY] [--max-tokens MAX_TOKENS] [--input-tokens INPUT_TOKENS] [--dataset DATASET] [--no-shuffle]
               [--seed SEED] [--timeout TIMEOUT] [-o OUTPUT] [-q] [-k]
               [task]

yabench - Benchmark OpenAI-compatible LLM APIs

positional arguments:
  task                  Task name from config file

options:
  -h, --help            show this help message and exit
  --config CONFIG       Config file path (default: yabench.yaml)
  --list                List available tasks from config
  --download [DATASET]  Download datasets. Use --download for all, or --download NAME for specific
  --base-url BASE_URL   Base URL for the OpenAI-compatible API
  --model MODEL         Model name (optional if API has default)
  --token TOKEN         Auth token (sent as X-Auth-Token header). Env: YABENCH_TOKEN
  --api-key API_KEY     API key (sent as Bearer token). Env: OPENAI_API_KEY
  -n NUM_REQUESTS, --num-requests NUM_REQUESTS
                        Number of requests to send (default: 10)
  -c CONCURRENCY, --concurrency CONCURRENCY
                        Number of concurrent requests (default: 1)
  --max-tokens MAX_TOKENS
                        Max tokens per response (default: 256)
  --input-tokens INPUT_TOKENS
                        Approximate input tokens per prompt (default: 100, ignored if --dataset used)
  --dataset DATASET     Dataset name or file path. Built-in: sample-en, sample-zh, alpaca-en, alpaca-zh, sharegpt, sharegpt-small, belle, firefly
  --no-shuffle          Don't shuffle dataset prompts
  --seed SEED           Random seed for reproducible prompt selection (default: 42)
  --timeout TIMEOUT     Request timeout in seconds (default: 120)
  -o OUTPUT, --output OUTPUT
                        Output file for JSON results
  -q, --quiet           Suppress progress output
  -k, --no-verify-ssl   Disable SSL certificate verification

Examples:
  # Download all datasets for offline use
  yabench --download

  # Run a task from config file
  yabench my-task

  # Run with dataset
  yabench --base-url http://localhost:8000/v1 --dataset sharegpt -n 100

  # List available tasks
  yabench --list
```