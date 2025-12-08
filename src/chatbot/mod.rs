//! # Chatbot - 模块化聊天机器人库
//!
//! `chatbot` 是一个功能完整的聊天机器人库，集成了：
//! - **LLM 集成**：支持 OpenAI 兼容的 API
//! - **记忆管理**：短期记忆和长期记忆（RAG）
//! - **记忆评估**：智能评估对话价值，按需保存
//! - **向量检索**：基于 PostgreSQL + pgvector 的语义检索

// 核心模块
mod chat;
mod config;
mod llm;
mod memory;
mod memory_evaluation;
mod prompt_template;
mod rag;
mod rag_database;

// 公开导出
pub use chat::{ChatBot, ChatStats};
pub use config::{
    load_config, save_config, Config, DbConfig, EmbeddingConfig, LlmConfig, MemoryConfig,
    MemoryEvaluationConfig, PostgresConfig, RagConfig,
};
pub use memory_evaluation::{MemoryEvaluator, RetentionDuration};
pub use rag::TemporalMemory;

// 错误类型
pub use anyhow::{Error, Result};

/// 库版本
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}

