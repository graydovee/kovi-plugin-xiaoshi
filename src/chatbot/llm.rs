use async_openai::{config::OpenAIConfig, Client as OpenAIClient};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;

/// LLM 请求参数配置
#[derive(Debug, Clone, Default)]
pub struct LlmRequestParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
}

/// LLM 客户端封装
pub struct LlmClient {
    #[allow(dead_code)]
    openai_client: Option<OpenAIClient<OpenAIConfig>>,
    http_client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    request_params: LlmRequestParams,
}

/// 工具调用信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// 函数调用详情
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// LLM 响应消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl LlmMessage {
    /// 创建系统消息
    #[allow(dead_code)]
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// 创建用户消息
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// 创建助手消息
    #[allow(dead_code)]
    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// 创建带工具调用的助手消息
    pub fn assistant_with_tool_calls(content: Option<&str>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.map(|s| s.to_string()),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// 创建工具响应消息
    pub fn tool(content: &str, tool_call_id: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
        }
    }

    /// 从 (role, content) 元组创建消息
    pub fn from_tuple(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// LLM 完成响应
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl CompletionResponse {
    /// 检查是否有工具调用
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

impl LlmClient {
    /// 创建新的 LLM 客户端
    ///
    /// # 参数
    /// - `api_key`: API 密钥
    /// - `base_url`: API 基础 URL
    /// - `model`: 使用的模型名称
    /// - `request_params`: 请求参数配置
    pub fn new(
        api_key: String,
        base_url: String,
        model: String,
        request_params: LlmRequestParams,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let http_client = reqwest::Client::new();

        let config = OpenAIConfig::new()
            .with_api_key(api_key.clone())
            .with_api_base(base_url.clone());
        let openai_client = Some(OpenAIClient::with_config(config));

        Ok(Self {
            openai_client,
            http_client,
            api_key,
            base_url,
            model,
            request_params,
        })
    }

    /// 发送带历史记录的聊天请求（简单版本，不带工具）
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
        let llm_messages: Vec<LlmMessage> = messages
            .into_iter()
            .map(|(role, content)| LlmMessage::from_tuple(&role, &content))
            .collect();

        let response = self.chat_completion(llm_messages, None).await?;
        Ok(response.content.unwrap_or_default())
    }

    /// 发送带工具支持的聊天请求
    ///
    /// # 参数
    /// - `messages`: LLM 消息列表
    /// - `tools`: 可选的工具定义列表（OpenAI 格式）
    ///
    /// # 返回
    /// - `Ok(CompletionResponse)`: 包含内容和可能的工具调用
    /// - `Err`: 错误信息
    pub async fn chat_completion(
        &self,
        messages: Vec<LlmMessage>,
        tools: Option<&Vec<Value>>,
    ) -> Result<CompletionResponse, Box<dyn Error + Send + Sync>> {
        let url = if self.base_url.ends_with("/chat/completions") {
            self.base_url.clone()
        } else {
            format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
        };

        // 构建请求体
        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": messages,
        });

        // 添加可选的请求参数（仅在配置了的情况下添加，以兼容有限制的模型）
        if let Some(temp) = self.request_params.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = self.request_params.top_p {
            request_body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(max_tokens) = self.request_params.max_tokens {
            request_body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(presence_penalty) = self.request_params.presence_penalty {
            request_body["presence_penalty"] = serde_json::json!(presence_penalty);
        }
        if let Some(frequency_penalty) = self.request_params.frequency_penalty {
            request_body["frequency_penalty"] = serde_json::json!(frequency_penalty);
        }

        // 如果有工具，添加到请求中
        if let Some(tools) = tools {
            if !tools.is_empty() {
                request_body["tools"] = serde_json::json!(tools);
            }
        }

        let response = self
            .http_client
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
        let json: serde_json::Value = serde_json::from_str(&response_text).map_err(|e| {
            format!(
                "Failed to parse OpenAI response: {}. Body: {}",
                e, response_text
            )
        })?;

        // 检查是否有错误字段
        if let Some(error) = json.get("error") {
            return Err(format!("OpenAI API returned error: {}", error).into());
        }

        // 解析响应
        let choice = &json["choices"][0]["message"];

        let content = choice["content"].as_str().map(|s| s.to_string());

        let tool_calls = if let Some(calls) = choice["tool_calls"].as_array() {
            calls
                .iter()
                .filter_map(|call| {
                    Some(ToolCall {
                        id: call["id"].as_str()?.to_string(),
                        call_type: call["type"].as_str().unwrap_or("function").to_string(),
                        function: FunctionCall {
                            name: call["function"]["name"].as_str()?.to_string(),
                            arguments: call["function"]["arguments"].as_str()?.to_string(),
                        },
                    })
                })
                .collect()
        } else {
            vec![]
        };

        Ok(CompletionResponse {
            content,
            tool_calls,
        })
    }

    /// 获取当前使用的模型名称
    #[allow(dead_code)]
    pub fn model(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = LlmClient::new(
            "test-key".to_string(),
            "https://api.openai.com/v1".to_string(),
            "gpt-3.5-turbo".to_string(),
        )
        .unwrap();
        assert_eq!(client.model(), "gpt-3.5-turbo");
    }

    #[test]
    fn test_llm_message_creation() {
        let system = LlmMessage::system("You are a helpful assistant");
        assert_eq!(system.role, "system");
        assert_eq!(system.content.as_deref(), Some("You are a helpful assistant"));

        let user = LlmMessage::user("Hello");
        assert_eq!(user.role, "user");

        let assistant = LlmMessage::assistant("Hi there!");
        assert_eq!(assistant.role, "assistant");

        let tool = LlmMessage::tool("result", "call_123");
        assert_eq!(tool.role, "tool");
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_123"));
    }

    #[test]
    fn test_tool_call_deserialization() {
        let json = r#"{
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\": \"Beijing\"}"
            }
        }"#;

        let tool_call: ToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tool_call.id, "call_abc123");
        assert_eq!(tool_call.function.name, "get_weather");
    }
}
