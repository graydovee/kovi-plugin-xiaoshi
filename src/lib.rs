mod chatbot;

use kovi::PluginBuilder as plugin;
use kovi::MsgEvent;
use std::sync::Arc;
use crate::chatbot::{ChatBot, load_config};

#[kovi::plugin]
async fn main() {
    let bot = plugin::get_runtime_bot();
    let data_path = bot.get_data_path();
    let config_json_path = data_path.join("config.json");

    // 加载配置
    let config = match load_config(&config_json_path) {
        Ok(cfg) => {
            kovi::log::info!("✅ 成功加载配置: {:?}, config: {:?}", config_json_path, cfg);
            cfg
        }
        Err(e) => {
            kovi::log::error!("❌ 加载配置失败: {}, 使用默认配置", e);
            chatbot::Config::default()
        }
    };

    // 初始化聊天机器人
    let chatbot = match ChatBot::new(config).await {
        Ok(service) => {
            let stats = service.get_stats();
            kovi::log::info!("🚀 聊天机器人初始化成功");
            kovi::log::info!("   LLM: {} ({})", stats.llm_provider, stats.llm_model);
            kovi::log::info!("   RAG: {}", if stats.rag_enabled { "已启用" } else { "未启用" });
            Arc::new(service)
        }
        Err(e) => {
            kovi::log::error!("❌ 聊天机器人初始化失败: {}", e);
            return;
        }
    };

    // 消息处理
    plugin::on_msg(move |event| {
        let chatbot = Arc::clone(&chatbot);

        async move {
            // 检查消息是否发给机器人
            if !is_to_me(&event) {
                return;
            }

            // 提取消息文本
            let text = match event.borrow_text() {
                Some(t) => t,
                None => return,
            };

            kovi::log::info!("📩 收到消息: {}", text);

            // 获取用户信息
            let user_id = event.sender.user_id;
            let group_id = if event.is_group() {
                event.group_id
            } else {
                None
            };
            
            // 优先使用群名片，其次昵称，最后默认值
            let sender_name = event
                .sender.card.clone()
                .or_else(|| event.sender.nickname.clone())
                .unwrap_or_else(|| "未知用户".to_string());

            // 调用聊天机器人
            match chatbot.chat(user_id, group_id, text, &sender_name).await {
                Ok(response) => {
                    event.reply(&response);
                }
                Err(e) => {
                    kovi::log::error!("❌ 聊天失败: {}", e);
                    event.reply(&format!("抱歉，处理消息时出错: {}", e));
                }
            }
        }
    });
}

fn is_to_me(event: &Arc<MsgEvent>) -> bool {
    if event.is_private() {
        return true;
    }
    if event.is_group() {
        let self_id_str = event.self_id.to_string();
        for segment in event.message.iter() {
            if segment.type_ == "at" {
                if let Some(qq) = segment.data.get("qq") {
                    if qq.as_str() == Some(&self_id_str) {
                        return true;
                    }
                }
            }
        }
    }
    false
}
