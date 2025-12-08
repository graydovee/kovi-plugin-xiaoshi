use chrono::Local;
use crate::chatbot::rag::Dialogue;

/// 提示词模板构建器
pub struct PromptTemplate;

impl PromptTemplate {
    /// 构建完整的系统提示词
    /// 
    /// # 参数
    /// - `character_prompt`: 角色性格设置（来自 config）
    /// - `memories`: RAG 检索到的长期记忆
    /// - `max_memory_tokens`: 记忆部分的最大 token 数
    /// 
    /// # 返回
    /// 完整的系统提示词，包含：
    /// - 当前时间
    /// - 角色性格设置
    /// - 时间理解指引
    /// - 长期记忆（如果有）
    pub fn build_system_prompt(
        character_prompt: &str,
        memories: Option<&[Dialogue]>,
        max_memory_tokens: usize,
    ) -> String {
        let now = Local::now();
        let current_time = now.format("%Y-%m-%d %H:%M:%S 星期%w").to_string();
        
        let mut prompt = String::new();
        
        // 1. 时间上下文
        prompt.push_str(&format!("📅 当前时间：{}\n\n", current_time));
        
        // 2. 角色性格设置
        prompt.push_str("🎭 角色设定\n");
        prompt.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        prompt.push_str("下面扮演名为\"小诗\"的角色进行对话，你要时刻牢记自己的名字\n");
        prompt.push_str("接下来的对话回答请用纯文本，不要包含markdown等格式，也不要包含颜文字和emoji表情等其他非文本字符。\n\n");
        prompt.push_str(character_prompt);
        prompt.push_str("\n\n");
        
        // 3. 时间理解指引
        prompt.push_str("⏰ 时间理解指引\n");
        prompt.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        prompt.push_str("• 当用户提到\"今天\"、\"明天\"、\"昨天\"等时间词时，请基于当前时间理解\n");
        prompt.push_str("• 记忆中的时间戳帮助你判断信息是否过时\n");
        prompt.push_str("• 如果记忆距离现在超过1天，可以主动提及时间跨度\n");
        prompt.push_str("• 对于时间敏感的承诺或约定，请特别注意时间上下文\n\n");
        
        // 4. 长期记忆（如果有）
        if let Some(memories) = memories {
            if !memories.is_empty() {
                prompt.push_str("📚 相关记忆（长期记忆）\n");
                prompt.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
                prompt.push_str("以下是与当前对话相关的历史记忆，按时间顺序排列：\n\n");
                
                let mut total_tokens = 0;
                for dialogue in memories {
                    let tokens = dialogue.token_count.unwrap_or((dialogue.content.len() / 4) as i32) as usize;
                    
                    // 检查是否超过 token 限制
                    if total_tokens + tokens > max_memory_tokens {
                        prompt.push_str("...\n（更多记忆因长度限制已省略）\n");
                        break;
                    }
                    
                    let formatted = Self::format_memory_item(dialogue);
                    prompt.push_str(&formatted);
                    prompt.push('\n');
                    
                    total_tokens += tokens;
                }
                
                prompt.push_str("\n");
            }
        }
        
        // 5. 对话指引
        prompt.push_str("💬 对话指引\n");
        prompt.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        prompt.push_str("• 如果记忆中有相关信息，请自然地引用，但不要生硬地复述\n");
        prompt.push_str("• 如果用户问到之前聊过的内容，可以回忆并回答\n");
        prompt.push_str("• 如果记忆中的信息可能过时，请谨慎使用并适当提醒\n");
        prompt.push_str("• 保持对话自然流畅，记忆只是辅助，不要让用户感觉到明显的\"检索\"\n");
        
        prompt
    }
    
    /// 格式化单条记忆为文本
    fn format_memory_item(dialogue: &Dialogue) -> String {
        let local_time: chrono::DateTime<chrono::Local> = dialogue.created_at.into();
        let abs_time = local_time.format("%Y-%m-%d %H:%M:%S");
        let rel_time = Self::format_relative_time(dialogue.created_at);
        
        let name = dialogue
            .sender_name
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("未知");
        
        let role_emoji = match dialogue.role.as_str() {
            "user" => "👤",
            "assistant" => "🤖",
            _ => "❓",
        };
        
        format!(
            "{} [{}] ({}) {}({}): {}",
            role_emoji,
            abs_time,
            rel_time,
            dialogue.role,
            name,
            dialogue.content
        )
    }
    
    /// 计算相对时间
    fn format_relative_time(timestamp: chrono::DateTime<chrono::Utc>) -> String {
        use chrono::{Duration, Utc};
        
        let now = Utc::now();
        let duration = now.signed_duration_since(timestamp);
        
        if duration < Duration::zero() {
            "未来".to_string()
        } else if duration < Duration::minutes(1) {
            "刚才".to_string()
        } else if duration < Duration::hours(1) {
            format!("{}分钟前", duration.num_minutes())
        } else if duration < Duration::days(1) {
            format!("{}小时前", duration.num_hours())
        } else if duration < Duration::days(7) {
            format!("{}天前", duration.num_days())
        } else if duration < Duration::days(30) {
            format!("{}周前", duration.num_weeks())
        } else {
            format!("{}个月前", duration.num_days() / 30)
        }
    }
    
    /// 构建简化的系统提示词（不包含长期记忆）
    /// 用于 RAG 未启用或检索失败的情况
    pub fn build_simple_system_prompt(character_prompt: &str) -> String {
        let mut prompt = String::new();
        prompt.push_str("下面扮演名为\"小诗\"的角色进行对话，你要时刻牢记自己的名字\n");
        prompt.push_str("接下来的对话回答请用纯文本，不要包含markdown等格式，也不要包含颜文字和emoji表情等其他非文本字符。\n\n");
        prompt.push_str(character_prompt);
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    
    #[test]
    fn test_build_simple_system_prompt() {
        let character = "你是一个友好的AI助手。";
        let prompt = PromptTemplate::build_simple_system_prompt(character);
        
        assert!(prompt.contains("当前时间"));
        assert!(prompt.contains(character));
    }
    
    #[test]
    fn test_format_relative_time() {
        let now = Utc::now();
        
        // 刚才
        let just_now = now - Duration::seconds(30);
        assert_eq!(PromptTemplate::format_relative_time(just_now), "刚才");
        
        // 5分钟前
        let five_min_ago = now - Duration::minutes(5);
        assert_eq!(PromptTemplate::format_relative_time(five_min_ago), "5分钟前");
        
        // 2小时前
        let two_hours_ago = now - Duration::hours(2);
        assert_eq!(PromptTemplate::format_relative_time(two_hours_ago), "2小时前");
        
        // 3天前
        let three_days_ago = now - Duration::days(3);
        assert_eq!(PromptTemplate::format_relative_time(three_days_ago), "3天前");
    }
}

