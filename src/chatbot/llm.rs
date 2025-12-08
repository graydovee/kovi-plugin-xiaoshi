use async_openai::{
    config::OpenAIConfig,
    Client as OpenAIClient,
};
use serde::{Deserialize, Serialize};
use std::error::Error;

/// LLM 提供商类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    OpenAI,
    Claude,
}

impl Provider {
    /// 从字符串解析提供商类型
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Provider::OpenAI),
            "claude" | "anthropic" => Ok(Provider::Claude),
            _ => Err(format!("不支持的 LLM 提供商: {}", s)),
        }
    }
}

// Claude API 请求和响应结构
#[derive(Debug, Serialize)]
struct ClaudeMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ClaudeRequest {
    model: String,
    messages: Vec<ClaudeMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    content: Vec<ClaudeContent>,
}

#[derive(Debug, Deserialize)]
struct ClaudeContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

/// LLM 客户端封装
pub struct LlmClient {
    provider: Provider,
    #[allow(dead_code)]
    openai_client: Option<OpenAIClient<OpenAIConfig>>,
    http_client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl LlmClient {
    /// 创建新的 LLM 客户端
    /// 
    /// # 参数
    /// - `provider`: 提供商类型（"openai" 或 "claude"）
    /// - `api_key`: API 密钥
    /// - `base_url`: API 基础 URL
    /// - `model`: 使用的模型名称
    pub fn new(
        provider: &str,
        api_key: String,
        base_url: String,
        model: String,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let provider = Provider::from_str(provider)?;
        let http_client = reqwest::Client::new();

        let openai_client = match provider {
            Provider::OpenAI => {
                let config = OpenAIConfig::new()
                    .with_api_key(api_key.clone())
                    .with_api_base(base_url.clone());
                Some(OpenAIClient::with_config(config))
            }
            Provider::Claude => None,
        };

        Ok(Self {
            provider,
            openai_client,
            http_client,
            api_key,
            base_url,
            model,
        })
    }

    /// 发送带历史记录的聊天请求
    /// 
    /// # 参数
    /// - `messages`: 消息历史，每个元素为 (role, content) 元组
    ///   - role 可以是 "system", "user", "assistant"
    /// 
    /// # 返回
    /// - `Ok(String)`: AI 的回复内容
    /// - `Err`: 错误信息
    pub async fn chat_with_history(
        &self,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        match self.provider {
            Provider::OpenAI => self.chat_with_history_openai(messages).await,
            Provider::Claude => self.chat_with_history_claude(messages).await,
        }
    }

    /// OpenAI 带历史的聊天实现
    async fn chat_with_history_openai(
        &self,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        // 使用 reqwest 直接调用，以便更好地处理错误和日志
        let url = if self.base_url.ends_with("/chat/completions") {
            self.base_url.clone()
        } else {
            format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
        };

        let mut json_messages = Vec::new();
        for (role, content) in messages {
             json_messages.push(serde_json::json!({
                 "role": role,
                 "content": content
             }));
        }

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": json_messages,
            "temperature": 0.1, // 降低温度以获得更稳定的评估结果
            "max_tokens": 1000,
        });

        let response = self.http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
             let status = response.status();
             let text = response.text().await.unwrap_or_default();
             return Err(format!("OpenAI API Error: {} - {}", status, text).into());
        }

        let response_text = response.text().await?;
        
        // 解析 JSON
        let json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| format!("Failed to parse OpenAI response: {}. Body: {}", e, response_text))?;

        // 检查是否有错误字段
        if let Some(error) = json.get("error") {
            return Err(format!("OpenAI API returned error: {}", error).into());
        }

        if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
            Ok(content.to_string())
        } else {
            Err(format!("Invalid OpenAI response structure (no content found). Body: {}", response_text).into())
        }
    }

    /// Claude 带历史的聊天实现
    async fn chat_with_history_claude(
        &self,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        // 分离 system 消息和其他消息
        let mut system_message = None;
        let mut chat_messages = Vec::new();

        for (role, content) in messages {
            match role.as_str() {
                "system" => {
                    system_message = Some(content);
                }
                "user" | "assistant" => {
                    chat_messages.push(ClaudeMessage { role, content });
                }
                _ => {
                    return Err(format!("不支持的角色类型: {}", role).into());
                }
            }
        }

        let request_body = ClaudeRequest {
            model: self.model.clone(),
            messages: chat_messages,
            max_tokens: 2000,
            temperature: Some(0.7),
            system: system_message,
        };

        let response = self.http_client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        // 检查响应状态
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "未知错误".to_string());
            return Err(format!("Claude API 请求失败: {} - {}", status, error_text).into());
        }

        // 解析响应
        let response_body: ClaudeResponse = response.json().await?;

        // 提取文本内容
        for content in &response_body.content {
            if content.content_type == "text" {
                return Ok(content.text.clone());
            }
        }

        Err("API 响应中没有找到文本内容".into())
    }

    /// 获取当前使用的模型名称
    #[cfg(test)]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// 获取当前使用的提供商
    #[cfg(test)]
    pub fn provider(&self) -> &Provider {
        &self.provider
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_from_str() {
        assert_eq!(Provider::from_str("openai").unwrap(), Provider::OpenAI);
        assert_eq!(Provider::from_str("claude").unwrap(), Provider::Claude);
        assert_eq!(Provider::from_str("anthropic").unwrap(), Provider::Claude);
        assert!(Provider::from_str("invalid").is_err());
    }

    #[test]
    fn test_client_creation_openai() {
        let client = LlmClient::new(
            "openai",
            "test-key".to_string(),
            "https://api.openai.com/v1".to_string(),
            "gpt-3.5-turbo".to_string(),
        ).unwrap();
        assert_eq!(client.model(), "gpt-3.5-turbo");
        assert_eq!(client.provider(), &Provider::OpenAI);
    }

    #[test]
    fn test_client_creation_claude() {
        let client = LlmClient::new(
            "claude",
            "test-key".to_string(),
            "https://api.anthropic.com/v1".to_string(),
            "claude-3-sonnet-20240229".to_string(),
        ).unwrap();
        assert_eq!(client.model(), "claude-3-sonnet-20240229");
        assert_eq!(client.provider(), &Provider::Claude);
    }
}

