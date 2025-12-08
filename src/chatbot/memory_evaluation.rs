/// 记忆评估模块
/// 
/// 根据对话内容的重要性评估记忆价值，决定保存时长

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};

use crate::chatbot::config::MemoryEvaluationConfig;
use crate::chatbot::llm::LlmClient;

/// 记忆保留时长枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionDuration {
    /// 不保存到长期记忆
    None,
    /// 保留1周
    OneWeek,
    /// 保留1个月
    OneMonth,
    /// 永久保留
    Forever,
}

impl RetentionDuration {
    /// 根据评分决定保留时长
    /// 
    /// # 评分标准
    /// - 0-25分：噪音与废弃（纯闲聊、无意义内容）
    /// - 26-60分：短期任务/1周（一次性工具、知识问答）
    /// - 61-85分：中期状态/1月（近期状态、软偏好）
    /// - 86-100分：永久画像/永久（事实性信息、长期偏好）
    pub fn from_score(score: i32) -> Self {
        match score {
            0..=25 => RetentionDuration::None,
            26..=60 => RetentionDuration::OneWeek,
            61..=85 => RetentionDuration::OneMonth,
            86..=100 => RetentionDuration::Forever,
            _ => RetentionDuration::None, // 超出范围默认不保存
        }
    }

    /// 计算过期时间
    /// 
    /// # 返回
    /// - Some(DateTime): 具体过期时间
    /// - None: 永不过期
    pub fn calculate_expiry(&self) -> Option<DateTime<Utc>> {
        let now = Utc::now();
        match self {
            RetentionDuration::None => Some(now), // 立即过期
            RetentionDuration::OneDay => Some(now + Duration::days(1)),
            RetentionDuration::OneWeek => Some(now + Duration::weeks(1)),
            RetentionDuration::OneMonth => Some(now + Duration::days(30)),
            RetentionDuration::Forever => None, // 永不过期
        }
    }

    /// 转换为可读字符串
    pub fn as_str(&self) -> &'static str {
        match self {
            RetentionDuration::None => "不保存",
            RetentionDuration::OneDay => "1天",
            RetentionDuration::OneWeek => "1周",
            RetentionDuration::OneMonth => "1个月",
            RetentionDuration::Forever => "永久",
        }
    }
}

/// 记忆评估器
pub struct MemoryEvaluator {
    llm_client: LlmClient,
    system_prompt: String,
}

impl MemoryEvaluator {
    /// 创建新的记忆评估器
    pub fn new(config: MemoryEvaluationConfig) -> Result<Self> {
        let llm_client = LlmClient::new(
            "openai", // 使用OpenAI兼容API
            config.apikey.clone(),
            config.url.clone(),
            config.model.clone(),
        ).map_err(|e| anyhow::anyhow!("记忆评估器初始化失败: {}", e))?;

        Ok(Self {
            llm_client,
            system_prompt: config.prompt,
        })
    }

    /// 评估对话的记忆价值
    /// 
    /// # 参数
    /// - `user_message`: 用户消息
    /// - `assistant_message`: AI回复
    /// 
    /// # 返回
    /// - 评分（0-100）
    pub async fn evaluate(&self, user_message: &str, assistant_message: &str) -> Result<i32> {
        use tokio::time::{timeout, Duration as TokioDuration};
        
        // 构建评估内容
        let conversation = format!(
            "User: {}\nAssistant: {}",
            user_message,
            assistant_message
        );

        // 构建消息历史
        let messages = vec![
            ("system".to_string(), self.system_prompt.clone()),
            ("user".to_string(), conversation),
        ];

        // 调用 LLM，添加30秒超时
        let response = timeout(
            TokioDuration::from_secs(30),
            self.llm_client.chat_with_history(messages)
        )
        .await
        .map_err(|_| anyhow::anyhow!("评估API调用超时（>30秒）"))?
        .map_err(|e| anyhow::anyhow!("评估API调用失败: {}", e))?;

        log::debug!("🤖 模型回复: [{}]", response);

        let content = response.trim();
        
        // 1. 尝试解析 JSON
        // 清理可能存在的 Markdown 代码块标记
        let json_str = if let Some(start) = content.find("{") {
            if let Some(end) = content.rfind("}") {
                &content[start..=end]
            } else {
                content
            }
        } else {
            content
        };

        #[derive(serde::Deserialize)]
        struct EvalResponse {
            score: i32,
            #[allow(dead_code)]
            reason: Option<String>,
        }

        if let Ok(eval) = serde_json::from_str::<EvalResponse>(json_str) {
            let score = eval.score.clamp(0, 100);
            if let Some(reason) = eval.reason {
                log::debug!("📊 记忆评估：{} 分 -> {} (理由: {})", score, RetentionDuration::from_score(score).as_str(), reason);
            } else {
                log::debug!("📊 记忆评估：{} 分 -> {}", score, RetentionDuration::from_score(score).as_str());
            }
            return Ok(score);
        }

        // 2. 降级：尝试解析纯数字
        if let Ok(score) = content.parse::<i32>() {
            let score = score.clamp(0, 100);
            log::debug!("📊 记忆评估（纯数字）：{} 分 -> {}", score, RetentionDuration::from_score(score).as_str());
            return Ok(score);
        }
        
        // 3. 降级：尝试提取数字
        let numbers: String = content.chars().filter(|c| c.is_ascii_digit()).collect();
        if let Ok(score) = numbers.parse::<i32>() {
            let score = score.clamp(0, 100);
            log::debug!("📊 记忆评估（提取数字）：{} 分 -> {}", score, RetentionDuration::from_score(score).as_str());
            return Ok(score);
        }

        // 默认给中等分数
        log::warn!("⚠ 无法解析评估结果（响应: {}），使用默认分数 50", content);
        Ok(50)
    }

    /// 评估并决定保留时长
    /// 
    /// # 返回
    /// (评分, 保留时长, 过期时间)
    pub async fn evaluate_and_decide(
        &self,
        user_message: &str,
        assistant_message: &str,
    ) -> Result<(i32, RetentionDuration, Option<DateTime<Utc>>)> {
        let score = self.evaluate(user_message, assistant_message).await?;
        let duration = RetentionDuration::from_score(score);
        let expiry = duration.calculate_expiry();
        
        Ok((score, duration, expiry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用例结构
    #[derive(Debug)]
    struct EvaluationTestCase {
        name: &'static str,
        user_message: &'static str,
        assistant_message: &'static str,
        expected_score_min: i32,
        expected_score_max: i32,
        expected_duration: RetentionDuration,
    }

    /// 获取所有测试用例
    fn get_test_cases() -> Vec<EvaluationTestCase> {
        vec![
            // ====== 区间 A: [0-25] 噪音与废弃 ======
            EvaluationTestCase {
                name: "简单寒暄",
                user_message: "你好啊",
                assistant_message: "你好！今天过得怎么样？",
                expected_score_min: 0,
                expected_score_max: 25,
                expected_duration: RetentionDuration::None,
            },
            EvaluationTestCase {
                name: "简单确认",
                user_message: "明白了，收到",
                assistant_message: "好的，如果还有其他问题随时告诉我。",
                expected_score_min: 0,
                expected_score_max: 25,
                expected_duration: RetentionDuration::None,
            },
            EvaluationTestCase {
                name: "无意义情绪",
                user_message: "哈哈哈哈笑死我了",
                assistant_message: "看来是有什么很有趣的事情呢。",
                expected_score_min: 0,
                expected_score_max: 25,
                expected_duration: RetentionDuration::None,
            },

            // ====== 区间 B: [26-60] 短期任务 (保留1周) ======
            EvaluationTestCase {
                name: "代码Debug (一次性工具)",
                user_message: "这段 Python 代码报错 KeyError: 'data' 怎么修？",
                assistant_message: "你需要先检查字典中是否存在该键，或者使用 .get('data') 方法。",
                expected_score_min: 26,
                expected_score_max: 60,
                expected_duration: RetentionDuration::OneWeek,
            },
            EvaluationTestCase {
                name: "翻译请求 (一次性工具)",
                user_message: "把这句话翻译成英文：'时不我待'",
                assistant_message: "Time waits for no one.",
                expected_score_min: 26,
                expected_score_max: 60,
                expected_duration: RetentionDuration::OneWeek,
            },
            EvaluationTestCase {
                name: "菜谱查询 (具体知识)",
                user_message: "宫保鸡丁怎么做？",
                assistant_message: "准备鸡胸肉、花生米、干辣椒...",
                expected_score_min: 26,
                expected_score_max: 60,
                expected_duration: RetentionDuration::OneWeek,
            },

            // ====== 区间 C: [61-85] 中期状态与软偏好 (保留1月) ======
            EvaluationTestCase {
                name: "近期计划 (状态导向)",
                user_message: "我最近在准备考研，压力有点大",
                assistant_message: "考研确实是一场持久战，要注意劳逸结合...",
                expected_score_min: 61,
                expected_score_max: 85,
                expected_duration: RetentionDuration::OneMonth,
            },
            EvaluationTestCase {
                name: "技术栈偏好 (软习惯)",
                user_message: "以后代码示例尽量用 Python，我比较熟悉",
                assistant_message: "好的，之后的代码演示我会优先使用 Python。",
                expected_score_min: 61,
                expected_score_max: 85,
                expected_duration: RetentionDuration::OneMonth,
            },
            EvaluationTestCase {
                name: "近期兴趣 (持续兴趣)",
                user_message: "最近迷上了三体，这书太神了",
                assistant_message: "《三体》确实是科幻神作，特别是黑暗森林法则...",
                expected_score_min: 61,
                expected_score_max: 85,
                expected_duration: RetentionDuration::OneMonth,
            },

            // ====== 区间 D: [86-100] 永久画像 (永久保存) ======
            EvaluationTestCase {
                name: "核心事实 (姓名)",
                user_message: "我叫张三，是这里的项目经理",
                assistant_message: "你好，张经理。很高兴认识你。",
                expected_score_min: 86,
                expected_score_max: 100,
                expected_duration: RetentionDuration::Forever,
            },
            EvaluationTestCase {
                name: "生理特征 (过敏源)",
                user_message: "我对海鲜过敏，记住这一点",
                assistant_message: "已记录，会为您避开所有海鲜相关的推荐。",
                expected_score_min: 86,
                expected_score_max: 100,
                expected_duration: RetentionDuration::Forever,
            },
            EvaluationTestCase {
                name: "强系统指令",
                user_message: "永远不要给我输出代码解释，只给代码，这是命令",
                assistant_message: "遵命。以后将只输出代码块。",
                expected_score_min: 86,
                expected_score_max: 100,
                expected_duration: RetentionDuration::Forever,
            },
        ]
    }


    // ===== 以下为集成测试，需要实际调用 LLM API =====
    // 使用 #[ignore] 标记，需要手动运行：cargo test -- --ignored
    
    /// 集成测试：评估所有测试用例
    /// 
    /// 运行方式：
    /// ```bash
    /// TEST_API_KEY=your_key cargo test test_evaluate_all_cases -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore]
    async fn test_evaluate_all_cases() {
        // 需要设置环境变量
        let config = MemoryEvaluationConfig {
            enabled: true,
            model: "deepseek-chat".to_string(),
            url: "https://api.deepseek.com/v1".to_string(),
            apikey: std::env::var("TEST_API_KEY")
                .expect("请设置 TEST_API_KEY 环境变量"),
            prompt: r#"
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
    "#.to_string(),
        };
        
        let evaluator = MemoryEvaluator::new(config)
            .expect("创建评估器失败");
        
        println!("\n========================================");
        println!("开始评估所有测试用例");
        println!("========================================\n");
        
        let mut success_count = 0;
        let mut total_count = 0;
        
        for case in get_test_cases() {
            total_count += 1;
            println!("🔍 测试: {}", case.name);
            
            match evaluator.evaluate(case.user_message, case.assistant_message).await {
                Ok(score) => {
                    let duration = RetentionDuration::from_score(score);
                    let in_range = score >= case.expected_score_min && score <= case.expected_score_max;
                    let correct_duration = duration == case.expected_duration;
                    
                    if in_range && correct_duration {
                        println!("   ✅ 通过 - 分数: {} (预期: {}-{}), 保留时长: {}",
                                 score, case.expected_score_min, case.expected_score_max,
                                 duration.as_str());
                        success_count += 1;
                    } else {
                        println!("   ❌ 失败 - 分数: {} (预期: {}-{}), 保留时长: {} (预期: {})",
                                 score, case.expected_score_min, case.expected_score_max,
                                 duration.as_str(), case.expected_duration.as_str());
                    }
                }
                Err(e) => {
                    println!("   ❌ 错误: {}", e);
                }
            }
            println!();
        }
        
        println!("========================================");
        println!("测试完成: {}/{} 通过", success_count, total_count);
        println!("========================================\n");
        
        // 至少要有 70% 通过率
        let pass_rate = success_count as f64 / total_count as f64;
        assert!(
            pass_rate >= 0.7,
            "通过率太低: {:.1}% (需要至少 70%)",
            pass_rate * 100.0
        );
    }
}

