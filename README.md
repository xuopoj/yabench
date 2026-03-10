# yabench

OpenAI 兼容 LLM API 的性能测试工具。单一二进制文件，无需运行时环境。

## 安装

从 [GitHub Releases](https://github.com/xuopoj/yabench/releases/latest) 下载对应平台的二进制文件：

```bash
# Linux x86_64
curl -L https://github.com/xuopoj/yabench/releases/latest/download/yabench-linux-x86_64 -o yabench
chmod +x yabench

# Linux arm64
curl -L https://github.com/xuopoj/yabench/releases/latest/download/yabench-linux-arm64 -o yabench
chmod +x yabench

# macOS (Apple Silicon)
curl -L https://github.com/xuopoj/yabench/releases/latest/download/yabench-darwin-arm64 -o yabench
chmod +x yabench

# macOS (Intel)
curl -L https://github.com/xuopoj/yabench/releases/latest/download/yabench-darwin-x86_64 -o yabench
chmod +x yabench
```

从源码编译：

```bash
cargo build --release
```

## 快速开始

```bash
# 生成配置文件
yabench --init

# 测试接口连通性
yabench --base-url http://localhost:8000/v1 --chat "你好"

# 运行压测
yabench --base-url http://localhost:8000/v1 -n 100 -c 10
```

## 配置文件

```bash
yabench --init    # 在当前目录生成 yabench.yaml
yabench --list    # 列出配置中的任务
```

`yabench.yaml` 示例：

```yaml
auth:
  hw-iam:
    type: iam
    iam_url: https://iam.cn-north-4.myhuaweicloud.com
    domain: 账号名
    username: 用户名
    password: ${IAM_PASSWORD}
    project: cn-north-4

  openai:
    type: api_key
    api_key: ${OPENAI_API_KEY}

defaults:
  num_requests: 100
  concurrency: 10
  max_tokens: 256

tasks:
  my-endpoint:
    base_url: https://your-endpoint/v1
    model: your-model
    auth: hw-iam
    dataset: alpaca-zh

  openai:
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
    auth: openai
```

运行任务：

```bash
yabench my-endpoint
```

## 指标说明

| 指标 | 说明 |
|------|------|
| TTFT | 首 token 延迟 |
| ITL | token 间延迟（均值） |
| E2E | 端到端延迟 |
| Output TPS | 输出 token 吞吐量 |
| Prefill TPS | 输入 token 数 / TTFT |

百分位统计：P50、P75、P90、P95、P98、P99。

## 数据集

内置数据集已编译进二进制文件：

| 名称 | 数量 | 语言 |
|------|------|------|
| `sample-en` | 50 | 英文 |
| `sample-zh` | 50 | 中文 |
| `alpaca-en` | 500 | 英文 |
| `alpaca-zh` | 500 | 中文 |

远程数据集（按需下载）：

| 名称 | 来源 |
|------|------|
| `sharegpt` | ShareGPT Vicuna |
| `sharegpt-small` | ShareGPT GPT-4 |
| `belle` | BELLE 1M 中文 |
| `firefly` | Firefly 1.1M |

```bash
yabench --list-datasets                  # 列出所有数据集
yabench --download sharegpt              # 下载指定数据集
yabench --print-dataset alpaca-zh -n 5  # 预览数据集内容
```

也支持自定义数据集文件（JSONL、JSON、TXT）。

## 认证方式

| 类型 | 请求头 | 配置字段 |
|------|--------|----------|
| `api_key` | `Authorization: Bearer` | `api_key` |
| `iam` | `X-Auth-Token`（自动获取） | `iam_url`、`domain`、`username`、`password`、`project` |

## 结果输出

```bash
yabench my-task -o results.json   # JSON 格式
yabench my-task -o results.csv    # CSV 格式（追加写入）
```

## 完整参数

```
Usage: yabench [OPTIONS] [TASK]

Arguments:
  [TASK]  配置文件中的任务名

Options:
      --config <CONFIG>              配置文件路径（默认：yabench.yaml）
      --list                         列出配置中的所有任务
      --download [<DATASET>]         下载数据集，不指定名称则下载全部
      --base-url <BASE_URL>          API 地址
      --model <MODEL>                模型名称
      --token <TOKEN>                认证 Token（X-Auth-Token）。环境变量：YABENCH_TOKEN
      --api-key <API_KEY>            API Key（Bearer Token）。环境变量：OPENAI_API_KEY
  -n, --num-requests <NUM_REQUESTS>  请求总数
  -c, --concurrency <CONCURRENCY>    并发数
      --max-tokens <MAX_TOKENS>      最大输出 token 数
      --input-tokens <INPUT_TOKENS>  输入 token 近似数量
      --dataset <DATASET>            数据集名称或文件路径
      --no-shuffle                   不打乱数据集顺序
      --seed <SEED>                  随机种子 [默认: 42]
      --timeout <TIMEOUT>            请求超时时间（秒）
  -o, --output <OUTPUT>              结果输出文件（.json 或 .csv）
  -q, --quiet                        不显示进度输出
  -k, --no-verify-ssl                跳过 SSL 证书验证
      --debug                        打印请求/响应详情
      --print-dataset <DATASET>      预览数据集内容
      --list-datasets                列出所有可用数据集
      --init                         在当前目录生成 yabench.yaml
      --warmup <WARMUP>              预热请求数 [默认: 0]
      --retries <RETRIES>            失败重试次数 [默认: 0]
      --chat <PROMPT>                发送单条消息并流式输出响应
  -h, --help                         显示帮助信息
```
