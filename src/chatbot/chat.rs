use anyhow::Result;
use std::sync::Arc;

use crate::chatbot::config::Config;
use crate::chatbot::llm::LlmClient;
use crate::chatbot::memory::Memory;
use crate::chatbot::prompt_template::PromptTemplate;
use crate::chatbot::rag::TemporalMemory;
use crate::chatbot::memory_evaluation::MemoryEvaluator;

/// 聊天机器人
/// 封装所有聊天相关的逻辑，包括记忆管理、RAG、LLM调用、记忆评估等
pub struct ChatBot {
    llm: Arc<LlmClient>,
    short_term_memory: Arc<Memory>,
    long_term_memory: Option<Arc<TemporalMemory>>,
    memory_evaluator: Option<Arc<MemoryEvaluator>>,
    config: Arc<Config>,
}

impl ChatBot {
    /// 创建新的聊天机器人
    pub async fn new(config: Config) -> Result<Self> {
        // 初始化 LLM 客户端
        let llm = LlmClient::new(
            &config.llm.provider,
            config.llm.apikey.clone(),
            config.llm.url.clone(),
            config.llm.model.clone(),
        )
        .map_err(|e| anyhow::anyhow!("LLM 客户端初始化失败: {}", e))?;

        // 初始化短期记忆
        let short_term_memory = Memory::new(
            config.memory.history_limit,
            config.memory.history_timeout,
        );

        // 初始化长期记忆（RAG）
        let long_term_memory = if config.memory.rag.enabled {
            match TemporalMemory::new(
                config.db.postgres.clone(),
                config.memory.rag.embedding.clone(),
                config.memory.rag.clone(),
            )
            .await
            {
                Ok(rag) => {
                    log::info!(
                        "✅ RAG 长期记忆已启用，锚点数: {}, 窗口大小: {}",
                        config.memory.rag.top_n,
                        config.memory.rag.window_size
                    );
                    Some(Arc::new(rag))
                }
                Err(e) => {
                    log::error!("❌ RAG 初始化失败: {}", e);
                    log::warn!("   将降级使用短期记忆模式");
                    None
                }
            }
        } else {
            log::info!("⏸️  RAG 未启用");
            None
        };

        // 初始化记忆评估器
        let memory_evaluator = if config.memory.rag.enabled && config.memory.rag.memory_evaluation.enabled {
            match MemoryEvaluator::new(config.memory.rag.memory_evaluation.clone()) {
                Ok(evaluator) => {
                    log::info!("✅ 记忆评估系统已启用");
                    Some(Arc::new(evaluator))
                }
                Err(e) => {
                    log::error!("❌ 记忆评估器初始化失败: {}", e);
                    log::warn!("   将使用默认保存策略");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            llm: Arc::new(llm),
            short_term_memory: Arc::new(short_term_memory),
            long_term_memory,
            memory_evaluator,
            config: Arc::new(config),
        })
    }

    /// 处理用户消息并返回AI回复
    ///
    /// # 参数
    /// - `user_id`: 用户QQ号
    /// - `group_id`: 群号（None表示私聊）
    /// - `user_input`: 用户输入文本
    /// - `sender_name`: 发送者昵称
    ///
    /// # 返回
    /// AI的回复文本
    pub async fn chat(
        &self,
        user_id: i64,
        group_id: Option<i64>,
        user_input: &str,
        sender_name: &str,
    ) -> Result<String> {
        let conversation_key = Memory::generate_key(user_id, group_id);

        // 步骤1: 如果启用了数据库，且短期记忆未初始化，则先初始化短期记忆
        if !self.short_term_memory.is_initialized(&conversation_key) {
            if let Some(rag) = &self.long_term_memory {
                if let Ok(recent_msgs) = rag
                    .get_recent_messages(user_id, group_id, self.config.memory.history_limit)
                    .await
                {
                    if !recent_msgs.is_empty() {
                        let messages: Vec<(String, String, String, u64)> = recent_msgs
                            .iter()
                            .map(|d| {
                                let timestamp = d.created_at.timestamp() as u64;
                                (
                                    d.message_uuid.clone(),
                                    d.role.clone(),
                                    d.content.clone(),
                                    timestamp,
                                )
                            })
                            .collect();

                        let count = self
                            .short_term_memory
                            .initialize_from_database(&conversation_key, messages);
                        if count > 0 {
                            log::info!("📚 从数据库加载 {} 条历史消息", count);
                        }
                    }
                }
            }
        }

        // 步骤2: 获取短期记忆的ID列表（用于后续去重）
        let short_term_ids = self.short_term_memory.get_message_ids(&conversation_key);

        // 步骤3: 检索长期记忆（排除短期记忆）
        let long_term_memories = if self.long_term_memory.is_some() {
            let rag = self.long_term_memory.as_ref().unwrap();

            // 检索长期记忆（排除短期记忆）
            match rag
                .get_contextual_memory(
                    user_id,
                    user_input,
                    group_id,
                    Some(self.config.memory.rag.top_n),
                    Some(self.config.memory.rag.window_size),
                    Some(&short_term_ids),
                )
                .await
            {
                Ok(memories) => {
                    if !memories.is_empty() {
                        log::info!("🔍 检索到 {} 条长期记忆", memories.len());
                    }
                    Some(memories)
                }
                Err(e) => {
                    log::warn!("⚠️  长期记忆检索失败: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // 步骤4: 使用长期记忆构建system prompt
        let system_prompt = if let Some(ref memories) = long_term_memories {
            if !memories.is_empty() {
                PromptTemplate::build_system_prompt(
                    &self.config.memory.prompt,
                    Some(memories),
                    self.config.memory.rag.max_memory_tokens,
                )
            } else {
                PromptTemplate::build_system_prompt(
                    &self.config.memory.prompt,
                    None,
                    self.config.memory.rag.max_memory_tokens,
                )
            }
        } else {
            PromptTemplate::build_simple_system_prompt(&self.config.memory.prompt)
        };

        // 步骤5: 使用prompt, 短期记忆，当前用户问题构建对话历史
        // 临时添加当前用户消息到历史中（仅用于构建prompt，不持久化）
        let mut history = self
            .short_term_memory
            .get_history(&conversation_key, &system_prompt);
        
        // 添加当前用户输入（临时的，用于LLM请求）
        history.push(("user".to_string(), user_input.to_string()));

        log::info!(
            "💭 对话 key: {}, 短期记忆: {} 条, 当前问题: 1 条",
            conversation_key,
            history.len() - 2  // 减去 system prompt 和当前用户消息
        );

        // 步骤6: 请求LLM
        let response = self
            .llm
            .chat_with_history(history)
            .await
            .map_err(|e| anyhow::anyhow!("LLM API 调用失败: {}", e))?;

        log::info!("🤖 AI回复: {}", response);

        // 步骤7: LLM成功响应后，保存当前对话到短期记忆
        let user_message_id = self
            .short_term_memory
            .add_user_message(&conversation_key, user_input.to_string());
        
        let assistant_message_id = self
            .short_term_memory
            .add_assistant_message(&conversation_key, response.clone());

        // 步骤8: 使用memory_evaluator评估对话价值，按需存入长期记忆
        // 这一步异步执行，不阻塞回复
        self.evaluate_and_store_memory_async(
            user_input.to_string(),
            response.clone(),
            sender_name.to_string(),
            user_id,
            group_id,
            user_message_id,
            assistant_message_id,
        );

        Ok(response)
    }

    /// 异步评估并存储记忆
    fn evaluate_and_store_memory_async(
        &self,
        user_input: String,
        response: String,
        sender_name: String,
        user_id: i64,
        group_id: Option<i64>,
        user_message_id: String,
        assistant_message_id: String,
    ) {
        if let Some(rag) = &self.long_term_memory {
            let rag = rag.clone();
            let memory_evaluator = self.memory_evaluator.clone();
            
            tokio::spawn(async move {
                if let Some(evaluator) = memory_evaluator {
                    // 使用评估器评估对话价值
                    match evaluator.evaluate_and_decide(&user_input, &response).await {
                        Ok((score, duration, expires_at)) => {
                            use crate::chatbot::memory_evaluation::RetentionDuration;
                            
                            // 如果评分足够高，才保存到长期记忆
                            if duration != RetentionDuration::None {
                                log::info!(
                                    "📊 记忆评估：{} 分 -> 保留 {}",
                                    score,
                                    duration.as_str()
                                );

                                // 保存用户消息
                                if let Err(e) = rag
                                    .add_dialogue(
                                        user_message_id,
                                        user_id,
                                        "user",
                                        &user_input,
                                        group_id,
                                        Some(&sender_name),
                                        None,
                                        Some(score),
                                        expires_at,
                                    )
                                    .await
                                {
                                    log::warn!("⚠️  存储用户消息到长期记忆失败: {}", e);
                                }

                                // 保存AI回复
                                if let Err(e) = rag
                                    .add_dialogue(
                                        assistant_message_id,
                                        user_id,
                                        "assistant",
                                        &response,
                                        group_id,
                                        Some("小诗"),
                                        None,
                                        Some(score),
                                        expires_at,
                                    )
                                    .await
                                {
                                    log::warn!("⚠️  存储AI回复到长期记忆失败: {}", e);
                                }
                            } else {
                                log::info!("📊 记忆评估：{} 分 -> 不保存到长期记忆", score);
                            }
                        }
                        Err(e) => {
                            log::warn!("⚠️  记忆评估失败: {}，使用默认策略保存（1周）", e);
                            
                            // 评估失败，使用默认策略保存（默认一周过期）
                            if let Err(e) = rag
                                .add_dialogue(
                                    user_message_id,
                                    user_id,
                                    "user",
                                    &user_input,
                                    group_id,
                                    Some(&sender_name),
                                    None,
                                    None,
                                    None,
                                )
                                .await
                            {
                                log::warn!("⚠️  存储用户消息到长期记忆失败: {}", e);
                            }

                            if let Err(e) = rag
                                .add_dialogue(
                                    assistant_message_id,
                                    user_id,
                                    "assistant",
                                    &response,
                                    group_id,
                                    Some("小诗"),
                                    None,
                                    None,
                                    None,
                                )
                                .await
                            {
                                log::warn!("⚠️  存储AI回复到长期记忆失败: {}", e);
                            }
                        }
                    }
                } else {
                    // 没有启用评估器，使用默认策略保存所有对话（默认一周过期）
                    if let Err(e) = rag
                        .add_dialogue(
                            user_message_id,
                            user_id,
                            "user",
                            &user_input,
                            group_id,
                            Some(&sender_name),
                            None,
                            None,
                            None,
                        )
                        .await
                    {
                        log::warn!("⚠️  存储用户消息到长期记忆失败: {}", e);
                    }

                    if let Err(e) = rag
                        .add_dialogue(
                            assistant_message_id,
                            user_id,
                            "assistant",
                            &response,
                            group_id,
                            Some("小诗"),
                            None,
                            None,
                            None,
                        )
                        .await
                    {
                        log::warn!("⚠️  存储AI回复到长期记忆失败: {}", e);
                    }
                }
            });
        }
    }

    /// 清除指定会话的历史
    #[allow(dead_code)]
    pub fn clear_history(&self, user_id: i64, group_id: Option<i64>) {
        let conversation_key = Memory::generate_key(user_id, group_id);
        self.short_term_memory.clear_history(&conversation_key);
        log::info!("🗑️  已清除会话 {} 的短期记忆", conversation_key);
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> ChatStats {
        ChatStats {
            conversation_count: self.short_term_memory.get_conversation_count(),
            rag_enabled: self.long_term_memory.is_some(),
            llm_provider: self.config.llm.provider.clone(),
            llm_model: self.config.llm.model.clone(),
        }
    }
    
    /// 清理过期记忆
    /// 
    /// 根据expires_at字段清理已过期的记忆
    /// 
    /// # 返回
    /// 清理的记录数量
    #[allow(dead_code)]
    pub async fn cleanup_expired_memories(&self) -> Result<u64> {
        if let Some(rag) = &self.long_term_memory {
            rag.cleanup_expired_memories().await
        } else {
            Ok(0)
        }
    }
}

/// 聊天统计信息
#[derive(Debug)]
#[allow(dead_code)]
pub struct ChatStats {
    pub conversation_count: usize,
    pub rag_enabled: bool,
    pub llm_provider: String,
    pub llm_model: String,
}

