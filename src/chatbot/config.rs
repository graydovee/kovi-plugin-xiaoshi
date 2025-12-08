use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub db: DbConfig,
    pub memory: MemoryConfig,
    #[serde(default)]
    pub mcp: McpConfig,
}

/// MCP (Model Context Protocol) 配置
/// 
/// MCP 配置文件格式示例 (mcp.json):
/// ```json
/// {
///   "mcpServers": {
///     "filesystem": {
///       "transport": "stdio",
///       "command": "npx",
///       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
///       "env": {}
///     },
///     "sse-server": {
///       "transport": "sse",
///       "url": "http://localhost:3000/sse"
///     },
///     "http-server": {
///       "transport": "streamable-http",
///       "url": "http://localhost:3000/mcp"
///     }
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// 是否启用 MCP
    #[serde(default)]
    pub enabled: bool,
    /// MCP 配置文件路径（标准 MCP 配置格式）
    #[serde(default)]
    pub path: String,
    /// 最大工具调用循环次数，防止无限循环
    #[serde(default = "default_max_tool_iterations")]
    pub max_tool_iterations: usize,
}

fn default_max_tool_iterations() -> usize {
    10
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            path: String::new(),
            max_tool_iterations: default_max_tool_iterations(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub model: String,
    pub url: String,
    pub apikey: String,
    /// 温度参数（0-2），控制输出的随机性，设为 None 使用 API 默认值
    #[serde(default)]
    pub temperature: Option<f64>,
    /// top_p 参数（0-1），控制采样范围，设为 None 使用 API 默认值
    #[serde(default)]
    pub top_p: Option<f64>,
    /// 最大输出 token 数，设为 None 使用 API 默认值
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// presence_penalty 参数（-2 到 2），设为 None 使用 API 默认值
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    /// frequency_penalty 参数（-2 到 2），设为 None 使用 API 默认值
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbConfig {
    pub postgres: PostgresConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresConfig {
    pub host: String,
    pub port: String,
    pub username: String,
    pub password: String,
    pub database: String,
    #[serde(default)]
    pub vector: VectorIndexConfig,
}

/// 向量索引配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexConfig {
    #[serde(default = "default_lists")]
    pub lists: usize,  // IVFFLAT 索引分区数
}

fn default_lists() -> usize {
    100
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            lists: default_lists(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub history_limit: usize,      // 历史记录条数限制
    pub history_timeout: u64,      // 历史记录超时时间（秒）
    #[serde(default = "default_prompt")]
    pub prompt: String,            // 系统提示词
    pub rag: RagConfig,            // RAG 配置
}

fn default_prompt() -> String {
    "你是一个可爱的虚拟女仆".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    #[serde(default)]
    pub enabled: bool,             // 是否启用 RAG（默认 false）
    pub embedding: EmbeddingConfig,
    pub top_n: usize,              // 向量检索锚点数量
    pub window_size: usize,        // 每锚点上下文宽度
    pub max_memory_tokens: usize,  // 记忆总token限制
    #[serde(default = "default_cleanup_days")]
    pub cleanup_days: u64,         // 清理过期数据的天数
    pub memory_evaluation: MemoryEvaluationConfig, // 记忆评估配置
}

fn default_cleanup_days() -> u64 {
    30
}

/// 记忆评估配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvaluationConfig {
    #[serde(default = "default_evaluation_enabled")]
    pub enabled: bool,             // 是否启用记忆评估（默认 true）
    pub model: String,             // 评估模型
    pub url: String,               // API URL
    pub apikey: String,            // API Key
    #[serde(default = "default_evaluation_prompt")]
    pub prompt: String,            // 评估提示词
    /// 温度参数（0-2），控制输出的随机性，设为 None 使用 API 默认值
    #[serde(default)]
    pub temperature: Option<f64>,
    /// top_p 参数（0-1），控制采样范围，设为 None 使用 API 默认值
    #[serde(default)]
    pub top_p: Option<f64>,
    /// 最大输出 token 数，设为 None 使用 API 默认值
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// presence_penalty 参数（-2 到 2），设为 None 使用 API 默认值
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    /// frequency_penalty 参数（-2 到 2），设为 None 使用 API 默认值
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
}

fn default_evaluation_enabled() -> bool {
    true
}

fn default_evaluation_prompt() -> String {
    r#"
### Role
你是一个RAG系统的记忆价值评估专家。你的任务是评估【用户与AI的对话】对未来交互的参考价值，并给出一个 0-100 的分数。

### 评分标准

#### 区间 A: [0-25] 噪音与废弃
**定义**：完全没有回溯价值的对话。
**包含**：
- 纯粹的礼貌寒暄 ("你好", "谢谢", "晚安")
- 简单的确认语 ("收到", "好的", "明白了")
- 情绪发泄与无意义字符 ("哈哈哈", "啊这", "测试123")
- **注意**：即使是用户说了话，如果没有包含任何实体信息或意图，也属于此类。

#### 区间 B: [26-60] 短期任务 (保留1周)
**定义**：动作导向。用户想要解决一个具体问题，或使用某种工具。
**包含**：
- **一次性工具使用**：翻译、润色文章、格式转换、代码Debug。
- **具体知识问答**：询问天气、百科知识、菜谱、旅游攻略。
- **逻辑**：这些信息在任务完成后（通常几天内）价值迅速衰减，但短期内有回溯必要。

#### 区间 C: [61-85] 中期状态与软偏好 (保留1月)
**定义**：状态导向 & 习惯导向。描述用户的近期状态、兴趣或可变的习惯。
**包含**：
- **近期状态**：正在进行的长期计划（"最近在减肥"、"正在准备考研"、"打算买房"）。
- **技术/风格偏好**：非绝对的习惯（"我喜欢用Python"、"文章写得幽默点"、"PPT用深色背景"）。
- **持续兴趣**：最近关注的话题（"最近迷上了三体"、"想学学炒股"）。

#### 区间 D: [86-100] 永久画像 (永久保存)
**定义**：身份导向。极难改变的事实与强指令。
**包含**：
- **核心事实**：姓名、性别、年龄、职业、居住地。
- **生理特征**：过敏源、残障信息（如色盲）。
- **强系统指令**：用户明确要求的永久性设定（"永远不要给我输出代码解释，只给代码"）。

### 输出格式 (JSON)
请严格输出合法的 JSON 格式，不要输出 Markdown 代码块标记：
{
    "score": 75,
    "reason": "用户提到了'喜欢用Python'，这属于技术栈偏好（软习惯），具有中长期的参考价值，归类为1月记忆。"
}
    "#.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model: String,
    pub url: String,
    pub apikey: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            llm: LlmConfig {
                model: String::new(),
                url: String::new(),
                apikey: String::new(),
                temperature: None,
                top_p: None,
                max_tokens: None,
                presence_penalty: None,
                frequency_penalty: None,
            },
            db: DbConfig {
                postgres: PostgresConfig {
                    host: "localhost".to_string(),
                    port: "5432".to_string(),
                    username: "postgres".to_string(),
                    password: String::new(),
                    database: "xiaoshi".to_string(),
                    vector: VectorIndexConfig::default(),
                },
            },
            memory: MemoryConfig {
                history_limit: 20,
                history_timeout: 600,
                prompt: default_prompt(),
                rag: RagConfig {
                    enabled: false,  // 默认不启用
                    embedding: EmbeddingConfig {
                        model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
                        url: "https://api.siliconflow.cn/v1/embeddings".to_string(),
                        apikey: String::new(),
                    },
                    top_n: 3,
                    window_size: 2,
                    max_memory_tokens: 1000,
                    cleanup_days: default_cleanup_days(),
                    memory_evaluation: MemoryEvaluationConfig {
                        enabled: default_evaluation_enabled(),
                        model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
                        url: "https://api.siliconflow.cn/v1".to_string(),
                        apikey: String::new(),
                        prompt: default_evaluation_prompt(),
                        temperature: None,
                        top_p: None,
                        max_tokens: None,
                        presence_penalty: None,
                        frequency_penalty: None,
                    },
                },
            },
            mcp: McpConfig::default(),
        }
    }
}

/// 加载配置文件
/// 如果配置文件不存在，会创建一个默认配置文件
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<Config, Box<dyn std::error::Error>> {
    let path = path.as_ref();
    
    // 如果配置文件不存在，创建默认配置
    if !path.exists() {
        let default_config = Config::default();
        save_config(path, &default_config)?;
        return Ok(default_config);
    }
    
    // 读取配置文件
    let content = fs::read_to_string(path)?;
    let config: Config = serde_json::from_str(&content)?;
    
    Ok(config)
}

/// 保存配置文件
pub fn save_config<P: AsRef<Path>>(path: P, config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let path = path.as_ref();
    
    // 确保父目录存在
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    // 将配置序列化为格式化的JSON
    let content = serde_json::to_string_pretty(config)?;
    fs::write(path, content)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.llm.model, deserialized.llm.model);
        assert_eq!(config.db.postgres.host, deserialized.db.postgres.host);
    }

    #[test]
    fn test_save_and_load_config() {
        let temp_path = "/tmp/test_config.json";
        
        // 创建测试配置
        let mut config = Config::default();
        config.llm.model = "gpt-4".to_string();
        config.llm.url = "https://api.openai.com".to_string();
        config.llm.apikey = "test-key".to_string();
        config.db.postgres.host = "localhost".to_string();
        
        // 保存配置
        save_config(temp_path, &config).unwrap();
        
        // 加载配置
        let loaded_config = load_config(temp_path).unwrap();
        
        assert_eq!(loaded_config.llm.model, "gpt-4");
        assert_eq!(loaded_config.llm.url, "https://api.openai.com");
        assert_eq!(loaded_config.llm.apikey, "test-key");
        assert_eq!(loaded_config.db.postgres.host, "localhost");
        
        // 清理测试文件
        fs::remove_file(temp_path).ok();
    }
}

