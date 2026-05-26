# yabench 使用指南

OpenAI 兼容 LLM API 性能测试工具完整使用指南。

## 目录

- [快速开始](#快速开始)
- [配置文件](#配置文件)
- [认证方式](#认证方式)
- [数据集](#数据集)
- [基础压测](#基础压测)
- [流式对话](#流式对话)
- [前缀缓存测试](#前缀缓存测试)
- [多轮对话回放](#多轮对话回放)
- [性能报告](#性能报告)
- [性能矩阵](#性能矩阵)
- [结果输出](#结果输出)
- [指标说明](#指标说明)
- [实用技巧](#实用技巧)

---

## 快速开始

```bash
# 生成配置文件模板
yabench --init

# 验证接口连通性（流式输出）
yabench --base-url http://localhost:8000/v1 --chat "你好"

# 运行基础压测：100 个请求，10 并发
yabench --base-url http://localhost:8000/v1 -n 100 -c 10

# 运行配置文件中的任务
yabench my-task
```

## 配置文件

运行 `yabench --init` 在当前目录生成 `yabench.yaml`。

### 文件结构

```yaml
auth:
  provider-name:
    type: api_key | iam
    # ... 认证相关字段

defaults:
  num_requests: 100
  concurrency: 10
  max_tokens: 256
  timeout: 120

tasks:
  task-name:
    base_url: https://your-endpoint/v1
    model: model-name
    auth: provider-name     # 引用上面定义的认证提供者
    # 任务级别覆盖...
```

### 核心特性

- **defaults 继承**：所有任务继承 `defaults` 中的配置，可在任务级别覆盖。
- **环境变量**：任何字符串值支持 `${ENV_VAR}` 语法。
- **大小后缀**：`max_tokens` 和 `input_tokens` 支持 `4K`、`32K`、`128K`、`1M` 写法。

### 任务管理

```bash
yabench --list             # 列出配置中的所有任务
yabench my-task            # 运行指定任务
yabench my-task -c 20      # 从命令行覆盖并发数
```

命令行参数始终优先于配置文件。

## 认证方式

### API Key（最常用）

```yaml
auth:
  openai:
    type: api_key
    api_key: ${OPENAI_API_KEY}
```

发送 `Authorization: Bearer <key>` 请求头。也可直接通过命令行传入：

```bash
yabench --base-url https://api.openai.com/v1 --api-key sk-... -n 10
```

### 华为云 IAM

```yaml
auth:
  hw:
    type: iam
    iam_url: https://iam.cn-north-4.myhuaweicloud.com
    domain: 账号名
    username: 用户名
    password: ${IAM_PASSWORD}
    project: cn-north-4
```

每次运行前自动获取 `X-Subject-Token`。

## 数据集

### 内置数据集（编译进二进制文件）

| 名称 | 数量 | 语言 |
|------|------|------|
| `sample-en` | 50 | 英文 |
| `sample-zh` | 50 | 中文 |
| `alpaca-en` | 500 | 英文 |
| `alpaca-zh` | 500 | 中文 |

### 远程数据集（首次使用自动下载）

| 名称 | 来源 |
|------|------|
| `sharegpt` | ShareGPT Vicuna（大型，多轮对话） |
| `sharegpt-small` | ShareGPT GPT-4（较小，多轮对话） |
| `belle` | BELLE 100万中文 |
| `firefly` | Firefly 110万中文 |

### 数据集命令

```bash
yabench --list-datasets                    # 列出所有可用数据集
yabench --download sharegpt                # 预下载远程数据集
yabench --download all                     # 下载全部
yabench --print-dataset alpaca-zh -n 5     # 预览前 5 条数据
```

### 使用数据集

```bash
# 在配置文件中指定
tasks:
  my-task:
    dataset: alpaca-zh

# 从命令行指定
yabench my-task --dataset alpaca-en
yabench --base-url http://localhost:8000/v1 --dataset /path/to/prompts.jsonl
```

支持的文件格式：JSONL、JSON 数组、纯文本（每行一个 prompt）。

支持的 JSONL 字段格式：
- `{"prompt": "..."}` 或 `{"text": "..."}` 或 `{"content": "..."}`
- `{"instruction": "...", "input": "..."}`
- `{"messages": [{"role": "user", "content": "..."}]}`
- `{"conversations": [{"from": "human", "value": "..."}]}`

## 基础压测

### 最简运行

```bash
yabench --base-url http://localhost:8000/v1 -n 50 -c 5
```

未指定数据集或 `--input-tokens` 时，默认使用 `sharegpt-small` 的多轮对话回放
（首次使用自动下载）。

### 控制 prompt 长度

```bash
# 将 prompt 填充到约 4K tokens（使用 ShareGPT 内容作为填充材料）
yabench my-task --input-tokens 4K

# 固定输出长度
yabench my-task --max-tokens 1K

# 长上下文测试
yabench my-task --input-tokens 128K --max-tokens 1K --timeout 600 -n 3 -c 1
```

启动时会自动校准 tokenizer 比率（向服务器发送一个短请求来测量 chars/token），
确保 `--input-tokens` 目标在任何模型的 tokenizer 下都足够准确。

### 预热和重试

```bash
yabench my-task --warmup 5        # 测量前发送 5 个预热请求
yabench my-task --retries 3       # 失败请求最多重试 3 次
```

### 可重复性

```bash
yabench my-task --seed 123        # 固定随机种子，数据集打乱顺序可重现
yabench my-task --no-shuffle      # 不打乱数据集顺序
```

## 流式对话

测试接口连通性，实时查看模型输出：

```bash
yabench my-task --chat "用三句话解释快速排序"
yabench --base-url http://localhost:8000/v1 --chat "你好"
```

token 到达时实时打印，结束后显示摘要：

```
[TTFT: 0.032s | tokens: 87 | 2.15s total]
```

## 前缀缓存测试

通过为每个请求添加相同的系统 prompt 前缀来测试服务端前缀缓存：

```bash
# 合成前缀（约 2000 tokens 的模板文本）
yabench my-task --prefix-tokens 2000

# 从文件加载自定义前缀
yabench my-task --prefix-file system-prompt.txt
```

所有请求共享相同的系统消息前缀。第一个请求填充服务端 KV 缓存后，后续请求的
TTFT 应明显降低（因为前缀命中缓存）。

### 观察重点

- 对比有无 `--prefix-tokens` 时的 TTFT 差异
- 缓存生效后，TTFT 应在预热完成后显著下降
- 使用 `--warmup 2` 在正式测量前预热缓存

```bash
# 无前缀缓存
yabench my-task -n 20 -c 1
# → TTFT p50: 0.150s

# 4K 前缀 + 预热
yabench my-task -n 20 -c 1 --prefix-tokens 4K --warmup 2
# → TTFT p50: 0.045s（4K 前缀命中缓存）
```

## 多轮对话回放

从 ShareGPT 格式数据集回放真实多轮对话。每段对话展开为递增前缀的请求序列：

```
第 1 轮: [user1]
第 2 轮: [user1, assistant1, user2]
第 3 轮: [user1, assistant1, user2, assistant2, user3]
```

这模拟了真实的多轮对话场景——每一轮复用前面所有轮次的内容作为前缀，
测试服务端前缀缓存在真实聊天负载下的表现。

```bash
# 默认使用 sharegpt-small 多轮回放
yabench my-task --multi-turn --dataset sharegpt-small

# 限制每段对话的回放轮数
yabench my-task --multi-turn --dataset sharegpt --max-turns-per-conversation 3
```

请求按轮次做跨对话的轮询发送（conv1.t1, conv2.t1, conv3.t1, conv1.t2, ...），
使并发请求命中不同对话的前缀——模拟真实场景中多个独立会话同时进行的流量模式。

## 性能报告

在相同 prompt 下梯度增加并发（c=1,2,4,8），寻找最优工作点：

```bash
yabench my-task --perf-report
yabench my-task --perf-report report.md
yabench my-task --perf-report --multi-turn --dataset sharegpt-small
yabench my-task --perf-report --prefix-tokens 4K
```

### 报告内容

生成的 Markdown 报告包含：
- **并发梯度表**：每个并发度下的 TTFT、E2E、TPS、ITL 各百分位
- **扩展性分析**：每次并发翻倍的吞吐增益和 TTFT 退化
- **推荐并发度**：自动检测吞吐量饱和拐点（增益低于 20% 的点）

### 缓存隔离

在每个并发级别之间，测试套件会运行一次**驱逐风暴**（8 个唯一随机 prompt，
每个约 4K tokens），通过 LRU 机制将上一级的缓存条目冲出，随后等待冷却。
这防止了上一级的热缓存条目泄漏到下一级，导致结果偏差。

## 性能矩阵

在**输入长度 × 输出长度 × 并发度**三个维度上做全面扫描——类似 ais_bench
的合成性能矩阵：

```bash
# 默认网格：5 输入 × 5 输出 × 3 并发 = 75 个格子
yabench my-task --perf-matrix

# 指定输出路径
yabench my-task --perf-matrix results.md

# 每格更多请求以获得更稳定的统计
yabench my-task --perf-matrix --matrix-n 20
```

### 默认网格

| 维度 | 取值 |
|------|------|
| 输入长度 | 1K, 4K, 16K, 64K, 128K |
| 输出长度 | 256, 1K, 4K, 16K, 64K |
| 并发度 | 1, 4, 8 |

### 自定义网格

```bash
yabench my-task --perf-matrix \
  --matrix-input 1K,4K,32K,128K \
  --matrix-output 256,1K,8K,64K \
  --matrix-concurrency 1,2,4,8,16
```

所有大小参数支持 K/M 后缀。总格子数 = 输入数 × 输出数 × 并发数。

### 输出格式

扁平 Markdown 表格，每个格子一行：

```
| Input | Output | c | Output TPS | Req/s | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | Prefill TPS | Errors |
|------:|-------:|--:|-----------:|------:|---------:|---------:|--------:|--------:|------------:|-------:|
|    1K |    256 |  1 |       45.2 |  7.96 |    0.032 |    0.041 |    5.68 |    6.12 |       31250 |      0 |
|    1K |    256 |  4 |      162.5 |  5.21 |    0.045 |    0.068 |    6.30 |    7.88 |       22222 |      0 |
...
```

### 注意事项

- 大格子（128K 输入 + 64K 输出）可能需要很长时间，确保 `--timeout` 足够大。
  完整默认网格建议设置 `--timeout 1800`。
- 快速摸底用 `--matrix-n 5`，稳定数据用 `--matrix-n 20`。
- prompt 使用 ShareGPT 语料填充到目标长度，tokenizer 比率按服务器自动校准。

## 结果输出

```bash
yabench my-task -o results.json    # JSON 格式（完整指标）
yabench my-task -o results.csv     # CSV 格式（追加写入）
```

CSV 追加模式适合将多次运行的结果收集到同一文件中做对比分析。

### JSON 结构

```json
{
  "num_requests": 100,
  "num_completed": 98,
  "num_errors": 2,
  "total_duration": 45.23,
  "throughput": {
    "requests_per_second": 2.17,
    "output_tps": 156.3,
    "prefill_tps": 8420.0
  },
  "latency": {
    "ttft": { "mean": 0.045, "p50": 0.038, "p75": 0.052, ... },
    "e2e":  { "mean": 12.3, "p50": 11.8, "p95": 15.2, ... },
    "inter_token_mean": 0.0064
  },
  "tokens": {
    "input": 245000,
    "output": 7068,
    "input_estimated_count": 0,
    "output_estimated_count": 0
  }
}
```

## 指标说明

| 指标 | 说明 |
|------|------|
| **TTFT** | 首 token 延迟（Time to First Token）——衡量 prefill 阶段延迟 |
| **E2E** | 端到端延迟——从发送请求到最后一个 token 的总耗时 |
| **ITL** | token 间延迟（Inter-Token Latency）——连续 token 之间的平均间隔 |
| **Output TPS** | 输出吞吐量——总输出 token 数 / 总耗时（聚合吞吐） |
| **Prefill TPS** | 预填充吞吐量——输入 token 数 / TTFT（单请求平均） |
| **Req/s** | 请求吞吐量——完成的请求数 / 总耗时 |

### 百分位统计

TTFT 报告 p50、p75、p90、p95、p98、p99；E2E 报告 p50、p95、p99。

小样本量（n < 50）时，p95 以上的百分位仅代表 1-2 个请求，可能不可靠。
如需可信的尾部延迟数据，建议至少使用 n=200。

### Token 计数

yabench 在每个请求中发送 `stream_options.include_usage: true`。大多数服务器
（vLLM、SGLang、OpenAI、MindIE）会返回精确的 token 计数。当服务器未返回
usage 时，yabench 会使用启动时校准的 chars/token 比率进行估算。估算的计数
会在结果摘要中标注。

## 实用技巧

### 快速连通性检查

```bash
yabench --base-url http://localhost:8000/v1 --chat "hi"
```

### 对比两个端点

```bash
yabench endpoint-a -n 100 -c 10 -o compare.csv
yabench endpoint-b -n 100 -c 10 -o compare.csv
# 两次结果追加到同一 CSV，方便对比
```

### 找到最优并发度

```bash
yabench my-task --perf-report
# 查看输出报告中的"Recommendation"章节
```

### 长上下文压力测试

```bash
yabench my-task --input-tokens 128K --max-tokens 1K -n 5 -c 1 --timeout 600
```

### 完整部署性能摸底

```bash
# 第一步：找到并发甜点
yabench my-task --perf-report

# 第二步：全面扫描不同输入/输出组合
yabench my-task --perf-matrix --matrix-n 20 --timeout 1800

# 第三步：测试前缀缓存效果
yabench my-task --perf-report --prefix-tokens 4K
```

### 静默模式（脚本集成）

```bash
yabench my-task -q -o results.json
```

### 调试连接问题

```bash
yabench my-task --chat "test" --debug
# 打印完整的请求/响应详情
```

### 跳过 SSL 验证

```bash
yabench my-task -k
# 或在配置文件中设置：no_verify_ssl: true
```
