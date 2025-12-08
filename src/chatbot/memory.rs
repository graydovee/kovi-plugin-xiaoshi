use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// 对话消息
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ChatMessage {
    pub message_id: String,  // 唯一消息ID（用于去重）
    pub role: String,        // "user" 或 "assistant"
    pub content: String,
    pub timestamp: u64,      // Unix 时间戳（保留用于未来功能）
}

/// 对话历史记录
#[derive(Debug, Clone)]
struct ConversationHistory {
    messages: Vec<ChatMessage>,
    last_update: u64,
}

/// 对话记忆管理器
pub struct Memory {
    histories: Arc<Mutex<HashMap<String, ConversationHistory>>>,
    history_limit: usize,
    history_timeout: u64,
}

impl Memory {
    /// 创建新的记忆管理器
    /// 
    /// # 参数
    /// - `history_limit`: 每个对话保留的最大消息数
    /// - `history_timeout`: 对话超时时间（秒），超时后清空历史
    pub fn new(history_limit: usize, history_timeout: u64) -> Self {
        Self {
            histories: Arc::new(Mutex::new(HashMap::new())),
            history_limit,
            history_timeout,
        }
    }

    /// 生成对话 key
    /// 
    /// # 参数
    /// - `user_id`: 用户 ID
    /// - `group_id`: 群组 ID（可选）
    /// 
    /// # 返回
    /// - 私聊：返回 "{user_id}"
    /// - 群聊：返回 "{group_id}:{user_id}"
    pub fn generate_key(user_id: i64, group_id: Option<i64>) -> String {
        match group_id {
            Some(gid) => format!("{}:{}", gid, user_id),
            None => user_id.to_string(),
        }
    }

    /// 获取当前时间戳
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
    
    /// 生成消息唯一ID
    /// 格式: {key}_{timestamp}_{role}_{random}
    fn generate_message_id(key: &str, timestamp: u64, role: &str) -> String {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hash, Hasher};
        
        let random_state = RandomState::new();
        let mut hasher = random_state.build_hasher();
        key.hash(&mut hasher);
        timestamp.hash(&mut hasher);
        role.hash(&mut hasher);
        
        format!("msg_{}_{}_{}_{:x}", key, timestamp, role, hasher.finish())
    }

    /// 添加用户消息
    /// 
    /// # 参数
    /// - `key`: 对话标识
    /// - `content`: 消息内容
    /// 
    /// # 返回
    /// 返回生成的消息ID
    pub fn add_user_message(&self, key: &str, content: String) -> String {
        let mut histories = self.histories.lock().unwrap();
        let timestamp = Self::current_timestamp();
        let message_id = Self::generate_message_id(key, timestamp, "user");

        let history = histories.entry(key.to_string()).or_insert_with(|| {
            ConversationHistory {
                messages: Vec::new(),
                last_update: timestamp,
            }
        });

        // 检查是否超时，如果超时则清空历史
        if timestamp - history.last_update > self.history_timeout {
            history.messages.clear();
        }

        // 添加用户消息
        history.messages.push(ChatMessage {
            message_id: message_id.clone(),
            role: "user".to_string(),
            content,
            timestamp,
        });

        // 限制历史消息数量（保留最近的消息）
        if history.messages.len() > self.history_limit {
            let excess = history.messages.len() - self.history_limit;
            history.messages.drain(0..excess);
        }

        history.last_update = timestamp;
        message_id
    }

    /// 添加 AI 回复消息
    /// 
    /// # 参数
    /// - `key`: 对话标识
    /// - `content`: 消息内容
    /// 
    /// # 返回
    /// 返回生成的消息ID
    pub fn add_assistant_message(&self, key: &str, content: String) -> String {
        let mut histories = self.histories.lock().unwrap();
        let timestamp = Self::current_timestamp();
        let message_id = Self::generate_message_id(key, timestamp, "assistant");

        if let Some(history) = histories.get_mut(key) {
            history.messages.push(ChatMessage {
                message_id: message_id.clone(),
                role: "assistant".to_string(),
                content,
                timestamp,
            });

            // 限制历史消息数量
            if history.messages.len() > self.history_limit {
                let excess = history.messages.len() - self.history_limit;
                history.messages.drain(0..excess);
            }

            history.last_update = timestamp;
        }
        
        message_id
    }

    /// 获取对话历史
    /// 
    /// # 参数
    /// - `key`: 对话标识
    /// - `system_prompt`: 系统提示词
    /// 
    /// # 返回
    /// 返回格式化的消息历史，包含 system 消息
    pub fn get_history(&self, key: &str, system_prompt: &str) -> Vec<(String, String)> {
        let mut histories = self.histories.lock().unwrap();
        let timestamp = Self::current_timestamp();

        let mut messages = vec![
            ("system".to_string(), system_prompt.to_string())
        ];

        if let Some(history) = histories.get_mut(key) {
            // 检查是否超时
            if timestamp - history.last_update > self.history_timeout {
                history.messages.clear();
                return messages;
            }

            // 添加历史消息
            for msg in &history.messages {
                messages.push((msg.role.clone(), msg.content.clone()));
            }
        }

        messages
    }
    
    /// 获取短期记忆的消息ID列表（用于去重）
    /// 
    /// # 参数
    /// - `key`: 对话标识
    /// 
    /// # 返回
    /// 返回短期记忆中所有消息的ID
    pub fn get_message_ids(&self, key: &str) -> Vec<String> {
        let histories = self.histories.lock().unwrap();
        
        if let Some(history) = histories.get(key) {
            history.messages.iter().map(|msg| msg.message_id.clone()).collect()
        } else {
            Vec::new()
        }
    }
    
    /// 从数据库初始化短期记忆
    /// 
    /// # 参数
    /// - `key`: 对话标识
    /// - `messages`: 从数据库加载的消息列表（按时间顺序）
    /// 
    /// # 返回
    /// 成功初始化的消息数量
    pub fn initialize_from_database(
        &self,
        key: &str,
        messages: Vec<(String, String, String, u64)>, // (message_id, role, content, timestamp)
    ) -> usize {
        let mut histories = self.histories.lock().unwrap();
        let timestamp = Self::current_timestamp();
        
        // 检查是否已经有历史记录
        if let Some(history) = histories.get(key) {
            if !history.messages.is_empty() {
                // 已有记忆，不覆盖
                return 0;
            }
        }
        
        // 创建或获取历史记录
        let history = histories.entry(key.to_string()).or_insert_with(|| {
            ConversationHistory {
                messages: Vec::new(),
                last_update: timestamp,
            }
        });
        
        // 添加消息
        let mut count = 0;
        for (message_id, role, content, msg_timestamp) in messages {
            history.messages.push(ChatMessage {
                message_id,
                role,
                content,
                timestamp: msg_timestamp,
            });
            count += 1;
        }
        
        // 限制数量
        if history.messages.len() > self.history_limit {
            let excess = history.messages.len() - self.history_limit;
            history.messages.drain(0..excess);
            count = self.history_limit;
        }
        
        history.last_update = timestamp;
        count
    }
    
    /// 检查是否已初始化
    pub fn is_initialized(&self, key: &str) -> bool {
        let histories = self.histories.lock().unwrap();
        histories.contains_key(key) && !histories.get(key).unwrap().messages.is_empty()
    }

    /// 清除指定对话的历史
    /// 
    /// # 参数
    /// - `key`: 对话标识
    #[allow(dead_code)]
    pub fn clear_history(&self, key: &str) {
        let mut histories = self.histories.lock().unwrap();
        histories.remove(key);
    }

    /// 清除所有对话历史
    #[allow(dead_code)]
    pub fn clear_all(&self) {
        let mut histories = self.histories.lock().unwrap();
        histories.clear();
    }

    /// 获取历史消息数量
    /// 
    /// # 参数
    /// - `key`: 对话标识
    #[allow(dead_code)]
    pub fn get_message_count(&self, key: &str) -> usize {
        let histories = self.histories.lock().unwrap();
        histories.get(key).map(|h| h.messages.len()).unwrap_or(0)
    }

    /// 获取所有对话数量
    #[allow(dead_code)]
    pub fn get_conversation_count(&self) -> usize {
        let histories = self.histories.lock().unwrap();
        histories.len()
    }

    /// 清理超时的对话历史
    #[allow(dead_code)]
    pub fn cleanup_expired(&self) {
        let mut histories = self.histories.lock().unwrap();
        let timestamp = Self::current_timestamp();

        histories.retain(|_, history| {
            timestamp - history.last_update <= self.history_timeout
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_key() {
        assert_eq!(Memory::generate_key(123456, None), "123456");
        assert_eq!(Memory::generate_key(123456, Some(789)), "789:123456");
    }

    #[test]
    fn test_add_and_get_messages() {
        let memory = Memory::new(10, 3600);
        let key = "test_user";

        memory.add_user_message(key, "你好".to_string());
        memory.add_assistant_message(key, "你好！有什么我可以帮你的吗？".to_string());

        let history = memory.get_history(key, "你是一个测试助手。");
        assert_eq!(history.len(), 3); // system + user + assistant
        assert_eq!(history[0].0, "system");
        assert_eq!(history[1].0, "user");
        assert_eq!(history[2].0, "assistant");
    }

    #[test]
    fn test_history_limit() {
        let memory = Memory::new(3, 3600);
        let key = "test_user";

        for i in 0..5 {
            memory.add_user_message(key, format!("消息 {}", i));
        }

        assert_eq!(memory.get_message_count(key), 3);
    }

    #[test]
    fn test_clear_history() {
        let memory = Memory::new(10, 3600);
        let key = "test_user";

        memory.add_user_message(key, "测试消息".to_string());
        assert_eq!(memory.get_message_count(key), 1);

        memory.clear_history(key);
        assert_eq!(memory.get_message_count(key), 0);
    }
}

