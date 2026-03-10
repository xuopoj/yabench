use anyhow::{anyhow, Result};
use reqwest::Client;
use serde_json::Value;
use std::time::Instant;
use tokio_stream::StreamExt;

#[derive(Debug, Clone)]
pub struct RequestMetrics {
    pub ttft: Option<f64>,
    pub total_time: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub inter_token_latencies: Vec<f64>,
    pub error: Option<String>,
}

impl RequestMetrics {
    pub fn new() -> Self {
        Self {
            ttft: None,
            total_time: 0.0,
            input_tokens: 0,
            output_tokens: 0,
            inter_token_latencies: vec![],
            error: None,
        }
    }

    pub fn tokens_per_second(&self) -> f64 {
        if self.total_time <= 0.0 || self.output_tokens == 0 {
            return 0.0;
        }
        self.output_tokens as f64 / self.total_time
    }

    pub fn prefill_tps(&self) -> f64 {
        match self.ttft {
            Some(ttft) if ttft > 0.0 && self.input_tokens > 0 => {
                self.input_tokens as f64 / ttft
            }
            _ => 0.0,
        }
    }

    pub fn mean_inter_token_latency(&self) -> f64 {
        if self.inter_token_latencies.is_empty() {
            return 0.0;
        }
        self.inter_token_latencies.iter().sum::<f64>() / self.inter_token_latencies.len() as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum StreamFormat {
    OpenAI,
    Mas,
}

fn detect_format(data: &str) -> StreamFormat {
    if let Ok(chunk) = serde_json::from_str::<Value>(data) {
        if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
            if let Some(first) = choices.first() {
                if first.get("delta").is_some() {
                    return StreamFormat::OpenAI;
                }
                if first.get("message").is_some() {
                    return StreamFormat::Mas;
                }
            }
        }
    }
    StreamFormat::OpenAI
}

struct ParsedChunk {
    content: Option<String>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    is_done: bool,
}

fn parse_line(line: &str, format: StreamFormat) -> Option<ParsedChunk> {
    // MAS usage events
    if format == StreamFormat::Mas && line.starts_with("event:") {
        let event_data = &line[6..];
        if let Ok(val) = serde_json::from_str::<Value>(event_data) {
            if let Some(usage) = val.get("usage") {
                return Some(ParsedChunk {
                    content: None,
                    input_tokens: usage.get("promptTokens").and_then(|v| v.as_u64()),
                    output_tokens: usage.get("completionTokens").and_then(|v| v.as_u64()),
                    is_done: false,
                });
            }
        }
        return None;
    }

    if !line.starts_with("data:") {
        return None;
    }

    let data = line[5..].trim_start();
    if data == "[DONE]" {
        return Some(ParsedChunk {
            content: None,
            input_tokens: None,
            output_tokens: None,
            is_done: true,
        });
    }

    let chunk: Value = serde_json::from_str(data).ok()?;

    let mut content = None;
    let mut input_tokens = None;
    let mut output_tokens = None;

    if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
        if let Some(first) = choices.first() {
            match format {
                StreamFormat::OpenAI => {
                    content = first
                        .get("delta")
                        .and_then(|d| d.get("content"))
                        .and_then(|c| c.as_str())
                        .map(String::from);
                }
                StreamFormat::Mas => {
                    content = first
                        .get("message")
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str())
                        .map(String::from);
                }
            }
        }
    }

    // OpenAI usage field in final chunk
    if let Some(usage) = chunk.get("usage") {
        input_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64());
        output_tokens = usage.get("completion_tokens").and_then(|v| v.as_u64());
    }

    Some(ParsedChunk {
        content,
        input_tokens,
        output_tokens,
        is_done: false,
    })
}

pub struct OpenAIClient {
    base_url: String,
    token: Option<String>,
    api_key: Option<String>,
    timeout_secs: f64,
    verify_ssl: bool,
    pub debug: bool,
}

impl OpenAIClient {
    pub fn new(
        base_url: String,
        token: Option<String>,
        api_key: Option<String>,
        timeout_secs: f64,
        verify_ssl: bool,
    ) -> Result<Self> {
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            token,
            api_key,
            timeout_secs,
            verify_ssl,
            debug: false,
        })
    }

    fn build_client(&self) -> Result<Client> {
        let mut builder = Client::builder()
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs));

        if !self.verify_ssl {
            builder = builder.danger_accept_invalid_certs(true);
        }

        Ok(builder.build()?)
    }

    /// Stream a single chat request, printing tokens to stdout as they arrive.
    /// Returns the full response text and final metrics.
    pub async fn chat(
        &self,
        prompt: &str,
        model: Option<&str>,
        max_tokens: u32,
    ) -> Result<(String, RequestMetrics)> {
        use std::io::Write;
        let mut metrics = RequestMetrics::new();
        let start = Instant::now();
        let mut full = String::new();

        let result: Result<()> = async {
            self.stream_complete_with(prompt, model, max_tokens, &mut metrics, start, |chunk| {
                full.push_str(chunk);
                print!("{chunk}");
                std::io::stdout().flush().ok();
            }).await
        }.await;

        metrics.total_time = start.elapsed().as_secs_f64();
        println!(); // newline after streamed output

        if let Err(e) = result {
            if metrics.error.is_none() {
                metrics.error = Some(e.to_string());
            }
        }

        Ok((full, metrics))
    }

    pub async fn complete(
        &self,
        prompt: &str,
        model: Option<&str>,
        max_tokens: u32,
    ) -> RequestMetrics {
        let mut metrics = RequestMetrics::new();
        let start = Instant::now();

        let result = self.stream_complete_with(prompt, model, max_tokens, &mut metrics, start, |_| {}).await;

        metrics.total_time = start.elapsed().as_secs_f64();

        if let Err(e) = result {
            if metrics.error.is_none() {
                metrics.error = Some(e.to_string());
            }
        }

        metrics
    }

    async fn stream_complete_with(
        &self,
        prompt: &str,
        model: Option<&str>,
        max_tokens: u32,
        metrics: &mut RequestMetrics,
        start: Instant,
        mut on_content: impl FnMut(&str),
    ) -> Result<()> {
        let client = self.build_client()?;

        let mut payload = serde_json::json!({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": true,
        });

        if let Some(m) = model {
            payload["model"] = Value::String(m.to_string());
        }

        let url = format!("{}/chat/completions", self.base_url);

        if self.debug {
            eprintln!("[debug] POST {url}");
            eprintln!("[debug] body: {}", serde_json::to_string_pretty(&payload).unwrap_or_default());
        }

        let mut req = client
            .post(&url)
            .header("Content-Type", "application/json");

        if let Some(token) = &self.token {
            req = req.header("X-Auth-Token", token);
        }
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let response = match req.json(&payload).send().await {
            Ok(r) => r,
            Err(e) => {
                let msg = if e.is_timeout() {
                    "timeout".to_string()
                } else {
                    format!("connection: {e}")
                };
                metrics.error = Some(msg.clone());
                return Err(anyhow!(msg));
            }
        };

        let status = response.status();
        if !status.is_success() {
            let code = status.as_u16();
            let body = response.text().await.unwrap_or_default();
            if self.debug {
                eprintln!("[debug] HTTP {code} response body:\n{body}");
            }
            let msg = format!("http_{code}: {}", body.chars().take(200).collect::<String>());
            metrics.error = Some(msg.clone());
            return Err(anyhow!(msg));
        }

        let mut stream = response.bytes_stream();
        let mut buf = String::new();
        let mut format: Option<StreamFormat> = None;
        let mut first_token = true;
        let mut last_token_time = start;
        // chunk count fallback if no usage in stream
        let mut chunk_count: u64 = 0;
        // track whether we got usage from the stream
        let mut got_usage = false;
        'outer: while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| anyhow!("stream error: {e}"))?;
            buf.push_str(&String::from_utf8_lossy(&bytes));

            // Process all complete lines
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim_end_matches('\r').to_string();
                buf.drain(..=pos);

                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                // Auto-detect format from first data line
                if format.is_none() && line.starts_with("data:") {
                    let data = line[5..].trim_start();
                    if data != "[DONE]" {
                        format = Some(detect_format(data));
                    }
                }

                let fmt = format.unwrap_or(StreamFormat::OpenAI);
                let parsed = match parse_line(line, fmt) {
                    Some(p) => p,
                    None => continue,
                };

                if parsed.is_done {
                    break 'outer;
                }

                // Update usage if provided
                if let Some(it) = parsed.input_tokens {
                    metrics.input_tokens = it;
                    got_usage = true;
                }
                if let Some(ot) = parsed.output_tokens {
                    metrics.output_tokens = ot;
                    got_usage = true;
                }

                // Handle content tokens
                if let Some(content) = &parsed.content {
                    if !content.is_empty() {
                        let now = Instant::now();
                        let elapsed = now.duration_since(start).as_secs_f64();

                        if first_token {
                            metrics.ttft = Some(elapsed);
                            first_token = false;
                        } else {
                            metrics.inter_token_latencies.push(
                                now.duration_since(last_token_time).as_secs_f64(),
                            );
                        }

                        last_token_time = now;
                        chunk_count += 1;
                        on_content(content);
                    }
                }
            }

        }

        // Fall back to chunk count if no usage from stream
        if !got_usage {
            metrics.output_tokens = chunk_count;
        }

        Ok(())
    }
}
