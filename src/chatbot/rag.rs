use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::chatbot::config::{EmbeddingConfig, PostgresConfig, RagConfig};
use crate::chatbot::rag_database::RagDatabase;

/// 对话消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dialogue {
    pub id: i32,
    pub message_uuid: String,    // 消息唯一标识（用于去重）
    pub user_id: i64,
    pub group_id: Option<i64>,
    pub chat_type: String,       // "private" 或 "group"
    pub role: String,            // "user" 或 "assistant"
    pub content: String,
    pub sender_name: Option<String>,
    pub qq_message_id: Option<i64>,  // QQ消息ID
    pub token_count: Option<i32>,
    pub score: Option<i32>,      // 记忆评分（0-100）
    pub expires_at: Option<DateTime<Utc>>,  // 过期时间
    pub created_at: DateTime<Utc>,
}

/// Embedding API 响应
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Embedding API 请求
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: String,
}

/// 时间感知的 RAG 记忆系统
pub struct TemporalMemory {
    database: RagDatabase,
    embedding_config: EmbeddingConfig,
    rag_config: RagConfig,
    http_client: reqwest::Client,
}

impl TemporalMemory {
    /// 创建新的 TemporalMemory 实例
    pub async fn new(
        postgres_config: PostgresConfig,
        embedding_config: EmbeddingConfig,
        rag_config: RagConfig,
    ) -> Result<Self> {
        // 创建数据库连接
        let database = RagDatabase::new(postgres_config).await?;

        Ok(Self {
            database,
            embedding_config,
            rag_config,
            http_client: reqwest::Client::new(),
        })
    }

    /// 生成会话标识
    /// 私聊："{user_id}"
    /// 群聊："{group_id}:{user_id}"
    pub fn generate_session_key(user_id: i64, group_id: Option<i64>) -> String {
        match group_id {
            Some(gid) => format!("{}:{}", gid, user_id),
            None => user_id.to_string(),
        }
    }

    /// 调用 Embedding API 获取向量
    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbeddingRequest {
            model: self.embedding_config.model.clone(),
            input: text.to_string(),
        };

        let response = self
            .http_client
            .post(&self.embedding_config.url)
            .header("Authorization", format!("Bearer {}", self.embedding_config.apikey))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow!("Embedding API 错误 [{}]: {}", status, body));
        }

        let embedding_response: EmbeddingResponse = response.json().await?;
        
        if embedding_response.data.is_empty() {
            return Err(anyhow!("Embedding API 返回空数据"));
        }

        Ok(embedding_response.data[0].embedding.clone())
    }

    /// 存储对话到长期记忆
    pub async fn add_dialogue(
        &self,
        message_uuid: String,
        user_id: i64,
        role: &str,
        content: &str,
        group_id: Option<i64>,
        sender_name: Option<&str>,
        qq_message_id: Option<i64>,
        score: Option<i32>,
        expires_at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<i32> {
        // 验证角色
        if role != "user" && role != "assistant" {
            return Err(anyhow!("角色必须是 'user' 或 'assistant'"));
        }

        // 确定聊天类型
        let chat_type = if group_id.is_some() { "group" } else { "private" };

        // 生成向量
        let embedding = self.get_embedding(content).await?;

        // 简单的 token 计数
        let token_count = (content.len() / 4) as i32;

        // 如果没有指定过期时间，默认一周后过期
        let expires_at = expires_at.or_else(|| Some(chrono::Utc::now() + chrono::Duration::weeks(1)));

        // 插入数据库
        self.database
            .insert_dialogue_with_score(
                &message_uuid,
                user_id,
                group_id,
                chat_type,
                role,
                content,
                sender_name,
                qq_message_id,
                &embedding,
                token_count,
                score,
                expires_at,
            )
            .await
    }

    /// 计算余弦相似度
    #[allow(dead_code)]
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let arr_a = Array1::from_vec(a.to_vec());
        let arr_b = Array1::from_vec(b.to_vec());

        let dot_product = arr_a.dot(&arr_b);
        let norm_a = arr_a.dot(&arr_a).sqrt();
        let norm_b = arr_b.dot(&arr_b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// 检索语义相关的记忆（带上下文窗口）
    pub async fn get_contextual_memory(
        &self,
        user_id: i64,
        query: &str,
        group_id: Option<i64>,
        top_n: Option<usize>,
        window_size: Option<usize>,
        exclude_message_ids: Option<&[String]>,
    ) -> Result<Vec<Dialogue>> {
        let top_n = top_n.unwrap_or(self.rag_config.top_n);
        let window_size = window_size.unwrap_or(self.rag_config.window_size);

        // 生成查询向量
        let query_embedding = self.get_embedding(query).await?;

        // 向量检索锚点
        let anchor_results = self
            .database
            .search_by_embedding(user_id, group_id, &query_embedding, exclude_message_ids, top_n)
            .await?;

        if anchor_results.is_empty() {
            return Ok(Vec::new());
        }

        // 收集锚点ID
        let anchor_ids: Vec<i32> = anchor_results.iter().map(|(id, _)| *id).collect();

        // 为每个锚点扩展上下文窗口
        let mut all_ids: Vec<i32> = Vec::new();
        for anchor_id in anchor_ids {
            let context_ids = self
                .database
                .get_context_window(user_id, group_id, anchor_id, window_size as i32)
                .await?;

            for id in context_ids {
                if !all_ids.contains(&id) {
                    all_ids.push(id);
                }
            }
        }

        // 去重并排序
        all_ids.sort();

        // 获取所有对话详情
        self.database.get_dialogues_by_ids(&all_ids).await
    }

    /// 批量插入历史对话（用于初始化）
    pub async fn bulk_insert_dialogues(&self, dialogues: Vec<Dialogue>) -> Result<usize> {
        let mut items = Vec::new();

        for dialogue in dialogues {
            // 生成向量
            let embedding = self.get_embedding(&dialogue.content).await?;

            items.push((
                dialogue.message_uuid,
                dialogue.user_id,
                dialogue.group_id,
                dialogue.chat_type,
                dialogue.role,
                dialogue.content,
                dialogue.sender_name,
                dialogue.qq_message_id,
                embedding,
                dialogue.token_count.unwrap_or(0),
                dialogue.created_at,
            ));
        }

        self.database.bulk_insert(items).await
    }

    /// 获取最近的对话（用于初始化短期记忆）
    pub async fn get_recent_messages(
        &self,
        user_id: i64,
        group_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<Dialogue>> {
        self.database.get_recent_messages(user_id, group_id, limit).await
    }
    
    /// 清理过期记忆
    pub async fn cleanup_expired_memories(&self) -> Result<u64> {
        self.database.cleanup_expired_memories().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_session_key() {
        assert_eq!(TemporalMemory::generate_session_key(123456, None), "123456");
        assert_eq!(
            TemporalMemory::generate_session_key(123456, Some(789)),
            "789:123456"
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((TemporalMemory::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!((TemporalMemory::cosine_similarity(&c, &d) - 0.0).abs() < 0.001);
    }
}

