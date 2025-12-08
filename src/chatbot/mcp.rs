//! MCP (Model Context Protocol) 客户端实现
//!
//! 支持三种传输模式：
//! - Stdio: 通过标准输入输出与 MCP 服务器通信
//! - SSE: 通过 Server-Sent Events 与 MCP 服务器通信
//! - StreamableHTTP: 通过 HTTP 流式传输与 MCP 服务器通信

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, Mutex, RwLock};

/// MCP 协议版本
pub const LATEST_PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC 请求 ID 生成器
static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

fn next_request_id() -> u64 {
    REQUEST_ID.fetch_add(1, Ordering::SeqCst)
}

// ============================================================================
// MCP 配置文件结构（标准格式）
// ============================================================================

/// 标准 MCP 配置文件结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfigFile {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServerConfig>,
}

/// MCP 服务器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "transport")]
pub enum McpServerConfig {
    /// Stdio 模式 - 通过进程的标准输入输出通信
    #[serde(rename = "stdio")]
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
    },
    /// SSE 模式 - 通过 Server-Sent Events 通信
    #[serde(rename = "sse")]
    Sse { url: String },
    /// StreamableHTTP 模式 - 通过 HTTP 流式传输通信
    #[serde(rename = "streamable-http")]
    StreamableHttp { url: String },
}

impl McpConfigFile {
    /// 从文件加载 MCP 配置
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow!("读取 MCP 配置文件失败: {}", e))?;
        let config: McpConfigFile = serde_json::from_str(&content)
            .map_err(|e| anyhow!("解析 MCP 配置文件失败: {}", e))?;
        Ok(config)
    }
}

// ============================================================================
// MCP 工具和内容类型
// ============================================================================

/// MCP 工具定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: McpToolInputSchema,
}

/// MCP 工具输入模式
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpToolInputSchema {
    #[serde(rename = "type", default)]
    pub schema_type: String,
    #[serde(default)]
    pub properties: Value,
    #[serde(default)]
    pub required: Vec<String>,
}

/// MCP 工具调用结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResult {
    #[serde(default)]
    pub content: Vec<McpContent>,
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

/// MCP 内容类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource { resource: Value },
}

// ============================================================================
// JSON-RPC 消息类型
// ============================================================================

/// JSON-RPC 请求
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<u64>,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

/// JSON-RPC 响应
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<u64>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

/// JSON-RPC 错误
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[allow(dead_code)]
    data: Option<Value>,
}

// ============================================================================
// MCP 传输层 trait
// ============================================================================

/// MCP 传输层 trait
#[async_trait::async_trait]
pub trait McpTransport: Send + Sync {
    /// 发送 JSON-RPC 请求并等待响应
    async fn send_request(&self, method: &str, params: Option<Value>) -> Result<Value>;
    /// 发送通知（不需要响应，不带 id）
    async fn send_notification(&self, method: &str) -> Result<()>;
    /// 关闭连接
    async fn close(&self);
}

// ============================================================================
// Stdio 传输实现
// ============================================================================

/// Stdio 传输
pub struct StdioTransport {
    stdin_tx: mpsc::Sender<String>,
    pending_requests: Arc<RwLock<HashMap<u64, tokio::sync::oneshot::Sender<Result<Value>>>>>,
    #[allow(dead_code)]
    child: Arc<Mutex<Option<Child>>>,
}

impl StdioTransport {
    /// 创建并启动 Stdio 传输
    pub async fn new(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
        server_name: &str,
    ) -> Result<Self> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        for (key, value) in env {
            cmd.env(key, value);
        }

        let mut child = cmd.spawn().map_err(|e| {
            anyhow!("启动 MCP 服务器 {} ({}) 失败: {}", server_name, command, e)
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("无法获取 stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("无法获取 stdout"))?;

        let (stdin_tx, mut stdin_rx) = mpsc::channel::<String>(100);

        // 启动写入任务
        let mut stdin_writer = stdin;
        tokio::spawn(async move {
            while let Some(msg) = stdin_rx.recv().await {
                if let Err(e) = stdin_writer.write_all(msg.as_bytes()).await {
                    log::error!("写入 MCP 服务器失败: {}", e);
                    break;
                }
                if let Err(e) = stdin_writer.flush().await {
                    log::error!("刷新 MCP 服务器输入失败: {}", e);
                    break;
                }
            }
        });

        let pending_requests: Arc<
            RwLock<HashMap<u64, tokio::sync::oneshot::Sender<Result<Value>>>>,
        > = Arc::new(RwLock::new(HashMap::new()));

        // 启动读取任务
        let pending_clone = pending_requests.clone();
        let name = server_name.to_string();
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                if line.is_empty() {
                    continue;
                }

                match serde_json::from_str::<JsonRpcResponse>(&line) {
                    Ok(response) => {
                        if let Some(id) = response.id {
                            let mut requests = pending_clone.write().await;
                            if let Some(tx) = requests.remove(&id) {
                                let result = if let Some(error) = response.error {
                                    Err(anyhow!(
                                        "MCP 错误 [{}]: {} (code: {})",
                                        name,
                                        error.message,
                                        error.code
                                    ))
                                } else {
                                    Ok(response.result.unwrap_or(Value::Null))
                                };
                                let _ = tx.send(result);
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("解析 MCP 响应失败: {} - 原始内容: {}", e, line);
                    }
                }
            }
        });

        log::info!("✅ MCP 服务器 {} (Stdio) 已启动", server_name);

        Ok(Self {
            stdin_tx,
            pending_requests,
            child: Arc::new(Mutex::new(Some(child))),
        })
    }
}

#[async_trait::async_trait]
impl McpTransport for StdioTransport {
    async fn send_request(&self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = next_request_id();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: method.to_string(),
            params,
        };

        let request_json = serde_json::to_string(&request)? + "\n";

        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut requests = self.pending_requests.write().await;
            requests.insert(id, tx);
        }

        self.stdin_tx
            .send(request_json)
            .await
            .map_err(|e| anyhow!("发送请求失败: {}", e))?;

        let result = tokio::time::timeout(std::time::Duration::from_secs(30), rx)
            .await
            .map_err(|_| anyhow!("MCP 请求超时"))?
            .map_err(|_| anyhow!("响应通道关闭"))??;

        Ok(result)
    }

    async fn send_notification(&self, method: &str) -> Result<()> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: method.to_string(),
            params: None,
        };

        let request_json = serde_json::to_string(&request)? + "\n";

        self.stdin_tx
            .send(request_json)
            .await
            .map_err(|e| anyhow!("发送通知失败: {}", e))?;

        Ok(())
    }

    async fn close(&self) {
        let mut child = self.child.lock().await;
        if let Some(mut c) = child.take() {
            let _ = c.kill().await;
        }
    }
}

// ============================================================================
// SSE 传输实现
// ============================================================================

/// SSE 传输
pub struct SseTransport {
    url: String,
    http_client: reqwest::Client,
    session_id: Arc<RwLock<Option<String>>>,
    pending_requests: Arc<RwLock<HashMap<u64, tokio::sync::oneshot::Sender<Result<Value>>>>>,
}

impl SseTransport {
    /// 创建 SSE 传输
    pub async fn new(url: &str, server_name: &str) -> Result<Self> {
        let http_client = reqwest::Client::new();
        let transport = Self {
            url: url.to_string(),
            http_client,
            session_id: Arc::new(RwLock::new(None)),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
        };

        // 启动 SSE 监听
        transport.start_sse_listener(server_name).await?;

        log::info!("✅ MCP 服务器 {} (SSE) 已连接", server_name);
        Ok(transport)
    }

    async fn start_sse_listener(&self, server_name: &str) -> Result<()> {
        let url = self.url.clone();
        let session_id = self.session_id.clone();
        let pending_requests = self.pending_requests.clone();
        let name = server_name.to_string();
        let client = self.http_client.clone();

        tokio::spawn(async move {
            use futures_util::StreamExt;
            
            loop {
                match client.get(&url).send().await {
                    Ok(response) => {
                        let mut stream = response.bytes_stream();

                        let mut buffer = String::new();
                        while let Some(chunk_result) = stream.next().await {
                            match chunk_result {
                                Ok(bytes) => {
                                    buffer.push_str(&String::from_utf8_lossy(&bytes));

                                    // 处理 SSE 事件
                                    while let Some(pos) = buffer.find("\n\n") {
                                        let event = buffer[..pos].to_string();
                                        buffer = buffer[pos + 2..].to_string();

                                        // 解析 SSE 事件
                                        if let Some(data) = event.strip_prefix("data: ") {
                                            if let Ok(response) =
                                                serde_json::from_str::<JsonRpcResponse>(data)
                                            {
                                                if let Some(id) = response.id {
                                                    let mut requests =
                                                        pending_requests.write().await;
                                                    if let Some(tx) = requests.remove(&id) {
                                                        let result =
                                                            if let Some(error) = response.error {
                                                                Err(anyhow!(
                                                                "MCP 错误 [{}]: {} (code: {})",
                                                                name,
                                                                error.message,
                                                                error.code
                                                            ))
                                                            } else {
                                                                Ok(response
                                                                    .result
                                                                    .unwrap_or(Value::Null))
                                                            };
                                                        let _ = tx.send(result);
                                                    }
                                                }
                                            }
                                        } else if let Some(sid) =
                                            event.strip_prefix("event: session\ndata: ")
                                        {
                                            let mut sess = session_id.write().await;
                                            *sess = Some(sid.trim().to_string());
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::error!("SSE 读取错误 [{}]: {}", name, e);
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("SSE 连接失败 [{}]: {}", name, e);
                    }
                }

                // 重连延迟
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        });

        // 等待会话建立
        for _ in 0..50 {
            if self.session_id.read().await.is_some() {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        Ok(()) // 即使没有会话ID也继续
    }

    fn get_post_url(&self) -> String {
        // SSE 模式下，POST 请求通常发送到不同的端点
        self.url.replace("/sse", "/message")
    }
}

#[async_trait::async_trait]
impl McpTransport for SseTransport {
    async fn send_request(&self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = next_request_id();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: method.to_string(),
            params,
        };

        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut requests = self.pending_requests.write().await;
            requests.insert(id, tx);
        }

        let post_url = self.get_post_url();
        let mut req = self.http_client.post(&post_url).json(&request);

        // 添加会话 ID
        if let Some(sid) = self.session_id.read().await.as_ref() {
            req = req.header("X-Session-Id", sid);
        }

        req.send()
            .await
            .map_err(|e| anyhow!("SSE 请求发送失败: {}", e))?;

        let result = tokio::time::timeout(std::time::Duration::from_secs(30), rx)
            .await
            .map_err(|_| anyhow!("MCP 请求超时"))?
            .map_err(|_| anyhow!("响应通道关闭"))??;

        Ok(result)
    }

    async fn send_notification(&self, method: &str) -> Result<()> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: method.to_string(),
            params: None,
        };

        let post_url = self.get_post_url();
        let mut req = self.http_client.post(&post_url).json(&request);

        if let Some(sid) = self.session_id.read().await.as_ref() {
            req = req.header("X-Session-Id", sid);
        }

        req.send()
            .await
            .map_err(|e| anyhow!("SSE 通知发送失败: {}", e))?;

        Ok(())
    }

    async fn close(&self) {
        // SSE 连接会在 drop 时自动关闭
    }
}

// ============================================================================
// StreamableHTTP 传输实现
// ============================================================================

/// StreamableHTTP 传输
pub struct StreamableHttpTransport {
    url: String,
    http_client: reqwest::Client,
    session_id: Arc<RwLock<Option<String>>>,
}

impl StreamableHttpTransport {
    /// 创建 StreamableHTTP 传输
    pub async fn new(url: &str, server_name: &str) -> Result<Self> {
        let http_client = reqwest::Client::new();

        log::info!("✅ MCP 服务器 {} (StreamableHTTP) 已连接", server_name);

        Ok(Self {
            url: url.to_string(),
            http_client,
            session_id: Arc::new(RwLock::new(None)),
        })
    }
}

#[async_trait::async_trait]
impl McpTransport for StreamableHttpTransport {
    async fn send_request(&self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = next_request_id();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: method.to_string(),
            params,
        };

        let mut req = self
            .http_client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&request);

        // 添加会话 ID
        if let Some(sid) = self.session_id.read().await.as_ref() {
            req = req.header("Mcp-Session-Id", sid);
        }

        let response = req
            .send()
            .await
            .map_err(|e| anyhow!("HTTP 请求发送失败: {}", e))?;

        // 保存会话 ID
        if let Some(sid) = response.headers().get("Mcp-Session-Id") {
            if let Ok(sid_str) = sid.to_str() {
                let mut sess = self.session_id.write().await;
                *sess = Some(sid_str.to_string());
            }
        }

        let content_type = response
            .headers()
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if content_type.contains("text/event-stream") {
            // 处理 SSE 响应
            let text = response.text().await?;
            
            // 按照 SSE 格式解析，事件由空行分隔
            let events: Vec<&str> = text.split("\n\n").collect();
            
            for event in events {
                let lines: Vec<&str> = event.lines().collect();
                let mut data_lines = Vec::new();
                
                for line in lines {
                    if let Some(data) = line.strip_prefix("data: ") {
                        data_lines.push(data);
                    } else if line.starts_with("data:") {
                        // 处理没有空格的情况
                        if let Some(data) = line.strip_prefix("data:") {
                            data_lines.push(data);
                        }
                    }
                }
                
                if !data_lines.is_empty() {
                    let data = data_lines.join("\n");
                    if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(&data) {
                        if let Some(error) = resp.error {
                            return Err(anyhow!("MCP 错误: {} (code: {})", error.message, error.code));
                        }
                        return Ok(resp.result.unwrap_or(Value::Null));
                    }
                }
            }
            Err(anyhow!("无法解析 SSE 响应"))
        } else {
            // 处理 JSON 响应
            let resp: JsonRpcResponse = response.json().await?;
            if let Some(error) = resp.error {
                return Err(anyhow!("MCP 错误: {} (code: {})", error.message, error.code));
            }
            Ok(resp.result.unwrap_or(Value::Null))
        }
    }

    async fn send_notification(&self, method: &str) -> Result<()> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: method.to_string(),
            params: None,
        };

        let mut req = self
            .http_client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&request);

        if let Some(sid) = self.session_id.read().await.as_ref() {
            req = req.header("Mcp-Session-Id", sid);
        }

        req.send()
            .await
            .map_err(|e| anyhow!("HTTP 通知发送失败: {}", e))?;

        Ok(())
    }

    async fn close(&self) {
        // HTTP 连接不需要显式关闭
    }
}

// ============================================================================
// MCP 客户端
// ============================================================================

/// MCP 客户端 - 支持多种传输模式
pub struct McpClient {
    name: String,
    transport: Box<dyn McpTransport>,
    tools: Arc<RwLock<Vec<McpTool>>>,
    initialized: Arc<Mutex<bool>>,
}

impl McpClient {
    /// 从配置创建 MCP 客户端
    pub async fn from_config(name: &str, config: &McpServerConfig) -> Result<Self> {
        let transport: Box<dyn McpTransport> = match config {
            McpServerConfig::Stdio { command, args, env } => {
                Box::new(StdioTransport::new(command, args, env, name).await?)
            }
            McpServerConfig::Sse { url } => Box::new(SseTransport::new(url, name).await?),
            McpServerConfig::StreamableHttp { url } => {
                Box::new(StreamableHttpTransport::new(url, name).await?)
            }
        };

        Ok(Self {
            name: name.to_string(),
            transport,
            tools: Arc::new(RwLock::new(Vec::new())),
            initialized: Arc::new(Mutex::new(false)),
        })
    }

    /// 初始化 MCP 连接
    pub async fn initialize(&self) -> Result<()> {
        let mut initialized = self.initialized.lock().await;
        if *initialized {
            return Ok(());
        }

        let init_params = json!({
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {
                "roots": { "listChanged": true },
                "sampling": {}
            },
            "clientInfo": {
                "name": "xiaoshi",
                "version": "1.0.0"
            }
        });

        let result = self
            .transport
            .send_request("initialize", Some(init_params))
            .await?;

        if let Some(server_info) = result.get("serverInfo") {
            let name = server_info
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let version = server_info
                .get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            log::info!("🔗 已连接到 MCP 服务器: {} v{}", name, version);
        }

        // 发送 initialized 通知（不需要响应，不带 id）
        let _ = self
            .transport
            .send_notification("notifications/initialized")
            .await;

        *initialized = true;
        Ok(())
    }

    /// 获取可用工具列表
    pub async fn list_tools(&self) -> Result<Vec<McpTool>> {
        let result = self.transport.send_request("tools/list", None).await?;

        let tools_value = result.get("tools").cloned().unwrap_or(Value::Array(vec![]));
        let tools: Vec<McpTool> = serde_json::from_value(tools_value)?;

        {
            let mut cached_tools = self.tools.write().await;
            *cached_tools = tools.clone();
        }

        for tool in &tools {
            log::info!("🔧 发现工具: {} - {}", tool.name, tool.description);
        }

        Ok(tools)
    }

    /// 调用工具
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<McpToolResult> {
        let params = json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.transport.send_request("tools/call", Some(params)).await?;
        let tool_result: McpToolResult = serde_json::from_value(result)?;

        Ok(tool_result)
    }

    /// 获取服务器名称
    #[allow(dead_code)]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 关闭客户端
    #[allow(dead_code)]
    pub async fn shutdown(&self) {
        self.transport.close().await;
        log::info!("🔌 MCP 服务器 {} 已关闭", self.name);
    }
}

// ============================================================================
// MCP 管理器
// ============================================================================

/// MCP 管理器 - 管理多个 MCP 客户端
pub struct McpManager {
    clients: HashMap<String, Arc<McpClient>>,
    tool_to_client: Arc<RwLock<HashMap<String, String>>>,
    all_tools: Arc<RwLock<Vec<McpTool>>>,
}

impl McpManager {
    /// 创建新的 MCP 管理器
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            tool_to_client: Arc::new(RwLock::new(HashMap::new())),
            all_tools: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// 从配置文件创建并初始化 MCP 管理器
    pub async fn from_config_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = McpConfigFile::load(path)?;
        Self::from_config(config).await
    }

    /// 从配置创建并初始化 MCP 管理器
    pub async fn from_config(config: McpConfigFile) -> Result<Self> {
        let mut manager = Self::new();

        for (name, server_config) in config.mcp_servers {
            match McpClient::from_config(&name, &server_config).await {
                Ok(client) => {
                    if let Err(e) = client.initialize().await {
                        log::error!("❌ 初始化 MCP 服务器 {} 失败: {}", name, e);
                        continue;
                    }
                    manager.clients.insert(name, Arc::new(client));
                }
                Err(e) => {
                    log::error!("❌ 创建 MCP 客户端 {} 失败: {}", name, e);
                    continue;
                }
            }
        }

        manager.refresh_tools().await?;
        Ok(manager)
    }

    /// 刷新所有工具列表
    pub async fn refresh_tools(&self) -> Result<()> {
        let mut all_tools = Vec::new();
        let mut tool_mapping = HashMap::new();

        for (name, client) in &self.clients {
            match client.list_tools().await {
                Ok(tools) => {
                    for tool in tools {
                        tool_mapping.insert(tool.name.clone(), name.clone());
                        all_tools.push(tool);
                    }
                }
                Err(e) => {
                    log::error!("❌ 获取 MCP 服务器 {} 的工具列表失败: {}", name, e);
                }
            }
        }

        {
            let mut cached = self.all_tools.write().await;
            *cached = all_tools;
        }
        {
            let mut mapping = self.tool_to_client.write().await;
            *mapping = tool_mapping;
        }

        Ok(())
    }

    /// 获取所有可用工具
    pub async fn get_all_tools(&self) -> Vec<McpTool> {
        self.all_tools.read().await.clone()
    }

    /// 调用工具
    pub async fn call_tool(&self, tool_name: &str, arguments: Value) -> Result<McpToolResult> {
        let client_name = {
            let mapping = self.tool_to_client.read().await;
            mapping
                .get(tool_name)
                .cloned()
                .ok_or_else(|| anyhow!("找不到工具 {} 对应的 MCP 服务器", tool_name))?
        };

        let client = self
            .clients
            .get(&client_name)
            .ok_or_else(|| anyhow!("MCP 客户端 {} 不存在", client_name))?;

        client.call_tool(tool_name, arguments).await
    }

    /// 将 MCP 工具转换为 OpenAI 兼容的工具格式
    pub async fn get_openai_tools(&self) -> Vec<Value> {
        let tools = self.all_tools.read().await;
        tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": tool.input_schema.schema_type,
                            "properties": tool.input_schema.properties,
                            "required": tool.input_schema.required
                        }
                    }
                })
            })
            .collect()
    }

    /// 检查是否有可用工具
    #[allow(dead_code)]
    pub async fn has_tools(&self) -> bool {
        !self.all_tools.read().await.is_empty()
    }

    /// 关闭所有客户端
    #[allow(dead_code)]
    pub async fn shutdown(&self) {
        for (_, client) in &self.clients {
            client.shutdown().await;
        }
    }
}

impl Default for McpManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_config_deserialization() {
        let json = r#"{
            "mcpServers": {
                "stdio-server": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-everything"],
                    "env": {}
                },
                "sse-server": {
                    "transport": "sse",
                    "url": "http://localhost:3000/sse"
                },
                "http-server": {
                    "transport": "streamable-http",
                    "url": "http://localhost:3000/mcp"
                }
            }
        }"#;

        let config: McpConfigFile = serde_json::from_str(json).unwrap();
        assert_eq!(config.mcp_servers.len(), 3);
        assert!(config.mcp_servers.contains_key("stdio-server"));
        assert!(config.mcp_servers.contains_key("sse-server"));
        assert!(config.mcp_servers.contains_key("http-server"));
    }

    #[test]
    fn test_mcp_tool_deserialization() {
        let json = r#"{
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"}
                },
                "required": ["arg1"]
            }
        }"#;

        let tool: McpTool = serde_json::from_str(json).unwrap();
        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.description, "A test tool");
    }
}
