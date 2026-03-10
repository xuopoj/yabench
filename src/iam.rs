use anyhow::{anyhow, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TokenResponse {
    token: TokenInfo,
}

#[derive(Debug, Deserialize)]
struct TokenInfo {
    expires_at: String,
}

pub struct IamToken {
    pub token: String,
    pub expires_at: String,
}

pub async fn fetch_token(
    iam_url: &str,
    domain: &str,
    username: &str,
    password: &str,
    project: &str,
    verify_ssl: bool,
) -> Result<IamToken> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .danger_accept_invalid_certs(!verify_ssl)
        .build()?;

    let body = serde_json::json!({
        "auth": {
            "identity": {
                "methods": ["password"],
                "password": {
                    "user": {
                        "domain": { "name": domain },
                        "name": username,
                        "password": password,
                    }
                }
            },
            "scope": {
                "project": { "name": project }
            }
        }
    });

    let url = format!("{}/v3/auth/tokens", iam_url.trim_end_matches('/'));
    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(anyhow!("IAM auth failed (HTTP {}): {}", status, text));
    }

    let token = response
        .headers()
        .get("X-Subject-Token")
        .ok_or_else(|| anyhow!("IAM response missing X-Subject-Token header"))?
        .to_str()?
        .to_string();

    let info: TokenResponse = response.json().await?;

    Ok(IamToken {
        token,
        expires_at: info.token.expires_at,
    })
}
