use anyhow::Result;
use chrono::{DateTime, Utc, NaiveDateTime};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use pgvector::Vector;

use crate::chatbot::config::{PostgresConfig, VectorIndexConfig};
use crate::chatbot::rag::Dialogue;

/// RAG 数据库操作类
pub struct RagDatabase {
    pool: PgPool,
    vector_config: VectorIndexConfig,
    vector_indexes_created: Arc<AtomicBool>,
}

impl RagDatabase {
    /// 创建新的数据库连接
    pub async fn new(postgres_config: PostgresConfig) -> Result<Self> {
        let connection_string = format!(
            "postgres://{}:{}@{}:{}/{}",
            postgres_config.username,
            postgres_config.password,
            postgres_config.host,
            postgres_config.port,
            postgres_config.database
        );

        let vector_config = postgres_config.vector.clone();

        let pool = PgPoolOptions::new()
            .max_connections(20)
            .connect(&connection_string)
            .await?;

        let indexes_created = Self::initialize_database(&pool, &vector_config).await?;

        log::info!("✅ 数据库初始化完成");
        if !indexes_created {
            log::info!("💡 提示：向量索引会在数据量达到 100 条后自动创建");
        }
        
        Ok(Self { 
            pool,
            vector_config,
            vector_indexes_created: Arc::new(AtomicBool::new(indexes_created)),
        })
    }

    async fn initialize_database(pool: &PgPool, vector_config: &VectorIndexConfig) -> Result<bool> {
        log::info!("📦 开始初始化数据库...");
        
        log::info!("   - 启用 pgvector 扩展");
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(pool)
            .await?;

        log::info!("   - 创建 dialogues 表");
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS dialogues (
                id SERIAL PRIMARY KEY,
                message_uuid TEXT UNIQUE NOT NULL,
                user_id BIGINT NOT NULL,
                group_id BIGINT,
                chat_type TEXT CHECK (chat_type IN ('private', 'group')),
                role TEXT CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sender_name TEXT,
                qq_message_id BIGINT,
                embedding VECTOR(1024),
                token_count INTEGER,
                score INTEGER,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                created_date DATE GENERATED ALWAYS AS (created_at::date) STORED
            )
            "#,
        )
        .execute(pool)
        .await?;

        log::info!("   - 创建索引");
        
        sqlx::query("CREATE UNIQUE INDEX IF NOT EXISTS idx_message_uuid ON dialogues (message_uuid)")
            .execute(pool)
            .await?;

        let mut indexes_created = true;
        
        let group_index_sql = format!(
            r#"CREATE INDEX IF NOT EXISTS idx_group_embedding ON dialogues 
            USING ivfflat (embedding vector_cosine_ops) WITH (lists={})
            WHERE group_id IS NOT NULL"#,
            vector_config.lists
        );
        
        match sqlx::query(&group_index_sql).execute(pool).await {
            Ok(_) => log::info!("   ✓ 群聊向量索引创建成功 (lists={})", vector_config.lists),
            Err(e) => {
                log::warn!("   ⚠ 群聊向量索引创建失败（表可能为空）: {}", e);
                indexes_created = false;
            }
        }

        let private_index_sql = format!(
            r#"CREATE INDEX IF NOT EXISTS idx_private_embedding ON dialogues 
            USING ivfflat (embedding vector_cosine_ops) WITH (lists={})
            WHERE group_id IS NULL"#,
            vector_config.lists
        );
        
        match sqlx::query(&private_index_sql).execute(pool).await {
            Ok(_) => log::info!("   ✓ 私聊向量索引创建成功 (lists={})", vector_config.lists),
            Err(e) => {
                log::warn!("   ⚠ 私聊向量索引创建失败（表可能为空）: {}", e);
                indexes_created = false;
            }
        }

        sqlx::query(r#"CREATE INDEX IF NOT EXISTS idx_group_time ON dialogues (group_id, user_id, id DESC) 
            WHERE group_id IS NOT NULL"#)
            .execute(pool).await?;

        sqlx::query(r#"CREATE INDEX IF NOT EXISTS idx_private_time ON dialogues (user_id, id DESC) 
            WHERE group_id IS NULL"#)
            .execute(pool).await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_chat_context ON dialogues (chat_type, user_id, group_id, created_at DESC)")
            .execute(pool).await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_created_at ON dialogues (created_at)")
            .execute(pool).await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_expires_at ON dialogues (expires_at) WHERE expires_at IS NOT NULL")
            .execute(pool).await?;

        Ok(indexes_created)
    }

    pub async fn insert_dialogue_with_score(
        &self,
        message_uuid: &str,
        user_id: i64,
        group_id: Option<i64>,
        chat_type: &str,
        role: &str,
        content: &str,
        sender_name: Option<&str>,
        qq_message_id: Option<i64>,
        embedding: &[f32],
        token_count: i32,
        score: Option<i32>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<i32> {
        let embedding_vec = Vector::from(embedding.to_vec());

        let row = sqlx::query(
                "INSERT INTO dialogues 
                (message_uuid, user_id, group_id, chat_type, role, content, sender_name, qq_message_id, embedding, token_count, score, expires_at, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
                ON CONFLICT (message_uuid) DO NOTHING
                RETURNING id",
            )
            .bind(message_uuid)
            .bind(user_id)
            .bind(group_id)
            .bind(chat_type)
            .bind(role)
            .bind(content)
            .bind(sender_name)
            .bind(qq_message_id)
            .bind(embedding_vec)
            .bind(token_count)
            .bind(score)
            .bind(expires_at)
            .fetch_optional(&self.pool)
            .await?;
        
        if !self.vector_indexes_created.load(Ordering::Relaxed) {
            let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM dialogues")
                .fetch_one(&self.pool).await.unwrap_or(0);
                
            if count >= 100 {
                log::info!("📊 数据量已达到 {} 条，开始自动创建向量索引...", count);
                if let Err(e) = self.try_create_vector_indexes().await {
                    log::warn!("⚠ 自动创建向量索引失败: {}", e);
                }
            }
        }

        if let Some(r) = row {
            Ok(r.get(0))
        } else {
            let id: i32 = sqlx::query_scalar("SELECT id FROM dialogues WHERE message_uuid = $1")
                .bind(message_uuid).fetch_one(&self.pool).await?;
            Ok(id)
        }
    }

    pub async fn search_by_embedding(
        &self,
        user_id: i64,
        group_id: Option<i64>,
        embedding: &[f32],
        exclude_message_ids: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(i32, String)>> {
        let embedding_vec = Vector::from(embedding.to_vec());
        
        let exclude_ids: Vec<&str> = exclude_message_ids
            .unwrap_or(&[]).iter().map(|s| s.as_str()).collect();
        
        let query_str = if group_id.is_some() {
             if !exclude_ids.is_empty() {
                "SELECT id, message_uuid FROM dialogues WHERE user_id = $1 AND group_id = $2 AND message_uuid != ALL($3)
                 ORDER BY embedding <-> $4 LIMIT $5"
             } else {
                "SELECT id, message_uuid FROM dialogues WHERE user_id = $1 AND group_id = $2
                 ORDER BY embedding <-> $3 LIMIT $4"
             }
        } else {
             if !exclude_ids.is_empty() {
                "SELECT id, message_uuid FROM dialogues WHERE user_id = $1 AND group_id IS NULL AND message_uuid != ALL($2)
                 ORDER BY embedding <-> $3 LIMIT $4"
             } else {
                "SELECT id, message_uuid FROM dialogues WHERE user_id = $1 AND group_id IS NULL
                 ORDER BY embedding <-> $2 LIMIT $3"
             }
        };
        
        let rows = if let Some(gid) = group_id {
            if !exclude_ids.is_empty() {
                sqlx::query(query_str).bind(user_id).bind(gid).bind(&exclude_ids)
                    .bind(embedding_vec).bind(limit as i64).fetch_all(&self.pool).await?
            } else {
                sqlx::query(query_str).bind(user_id).bind(gid)
                    .bind(embedding_vec).bind(limit as i64).fetch_all(&self.pool).await?
            }
        } else {
            if !exclude_ids.is_empty() {
                sqlx::query(query_str).bind(user_id).bind(&exclude_ids)
                    .bind(embedding_vec).bind(limit as i64).fetch_all(&self.pool).await?
            } else {
                sqlx::query(query_str).bind(user_id)
                    .bind(embedding_vec).bind(limit as i64).fetch_all(&self.pool).await?
            }
        };
        
        let mut results = Vec::new();
        for row in rows { results.push((row.get(0), row.get(1))); }
        Ok(results)
    }

    pub async fn get_context_window(
        &self, user_id: i64, group_id: Option<i64>, anchor_id: i32, window_size: i32,
    ) -> Result<Vec<i32>> {
        let query = if group_id.is_some() {
            "SELECT id FROM dialogues WHERE user_id = $1 AND group_id = $2
               AND id >= $3 - $4 AND id <= $3 + $4 ORDER BY id"
        } else {
            "SELECT id FROM dialogues WHERE user_id = $1 AND group_id IS NULL
               AND id >= $2 - $3 AND id <= $2 + $3 ORDER BY id"
        };
        
        let rows = if let Some(gid) = group_id {
            sqlx::query(query).bind(user_id).bind(gid).bind(anchor_id).bind(window_size)
                .fetch_all(&self.pool).await?
        } else {
             sqlx::query(query).bind(user_id).bind(anchor_id).bind(window_size)
                .fetch_all(&self.pool).await?
        };
        
        Ok(rows.iter().map(|row| row.get(0)).collect())
    }

    pub async fn get_dialogues_by_ids(&self, ids: &[i32]) -> Result<Vec<Dialogue>> {
        let rows = sqlx::query(
                "SELECT id, message_uuid, user_id, group_id, chat_type, role, content, 
                        sender_name, qq_message_id, token_count, score, expires_at, created_at
                 FROM dialogues WHERE id = ANY($1) ORDER BY created_at",
            ).bind(ids).fetch_all(&self.pool).await?;
        
        let mut dialogues = Vec::new();
        for row in rows {
            let created_at: DateTime<Utc> = match row.try_get("created_at") {
                Ok(val) => val,
                Err(_) => {
                    let naive: NaiveDateTime = row.get("created_at");
                    DateTime::from_naive_utc_and_offset(naive, Utc)
                }
            };
            
            let expires_at: Option<DateTime<Utc>> = match row.try_get("expires_at") {
                Ok(val) => val,
                Err(_) => match row.try_get::<Option<NaiveDateTime>, _>("expires_at") {
                    Ok(Some(naive)) => Some(DateTime::from_naive_utc_and_offset(naive, Utc)),
                    _ => None
                }
            };

            dialogues.push(Dialogue {
                id: row.get("id"), message_uuid: row.get("message_uuid"),
                user_id: row.get("user_id"), group_id: row.get("group_id"),
                chat_type: row.get("chat_type"), role: row.get("role"),
                content: row.get("content"), sender_name: row.get("sender_name"),
                qq_message_id: row.get("qq_message_id"), token_count: row.get("token_count"),
                score: row.try_get("score").ok(), expires_at, created_at,
            });
        }
        Ok(dialogues)
    }

    pub async fn get_recent_messages(
        &self, user_id: i64, group_id: Option<i64>, limit: usize,
    ) -> Result<Vec<Dialogue>> {
        let query = if group_id.is_some() {
            "SELECT id, message_uuid, user_id, group_id, chat_type, role, content, 
                    sender_name, qq_message_id, token_count, score, expires_at, created_at
             FROM dialogues WHERE user_id = $1 AND group_id = $2 ORDER BY created_at DESC LIMIT $3"
        } else {
            "SELECT id, message_uuid, user_id, group_id, chat_type, role, content, 
                    sender_name, qq_message_id, token_count, score, expires_at, created_at
             FROM dialogues WHERE user_id = $1 AND group_id IS NULL ORDER BY created_at DESC LIMIT $2"
        };
        
        let rows = if let Some(gid) = group_id {
            sqlx::query(query).bind(user_id).bind(gid).bind(limit as i64).fetch_all(&self.pool).await?
        } else {
            sqlx::query(query).bind(user_id).bind(limit as i64).fetch_all(&self.pool).await?
        };
        
        let mut dialogues = Vec::new();
        for row in rows {
            let created_at: DateTime<Utc> = match row.try_get("created_at") {
                Ok(val) => val,
                Err(_) => {
                    let naive: NaiveDateTime = row.get("created_at");
                    DateTime::from_naive_utc_and_offset(naive, Utc)
                }
            };
            
            let expires_at: Option<DateTime<Utc>> = match row.try_get("expires_at") {
                Ok(val) => val,
                Err(_) => match row.try_get::<Option<NaiveDateTime>, _>("expires_at") {
                    Ok(Some(naive)) => Some(DateTime::from_naive_utc_and_offset(naive, Utc)),
                    _ => None
                }
            };

            dialogues.push(Dialogue {
                id: row.get("id"), message_uuid: row.get("message_uuid"),
                user_id: row.get("user_id"), group_id: row.get("group_id"),
                chat_type: row.get("chat_type"), role: row.get("role"),
                content: row.get("content"), sender_name: row.get("sender_name"),
                qq_message_id: row.get("qq_message_id"), token_count: row.get("token_count"),
                score: row.try_get("score").ok(), expires_at, created_at,
            });
        }
        
        dialogues.reverse();
        Ok(dialogues)
    }

    pub async fn bulk_insert(
        &self,
        dialogues: Vec<(String, i64, Option<i64>, String, String, String, Option<String>, Option<i64>, Vec<f32>, i32, DateTime<Utc>)>,
    ) -> Result<usize> {
        let mut inserted = 0;

        for (message_uuid, user_id, group_id, chat_type, role, content, sender_name, qq_message_id, embedding, token_count, created_at) in dialogues {
            let embedding_vec = Vector::from(embedding);
            
            let result = sqlx::query(
                    "INSERT INTO dialogues 
                    (message_uuid, user_id, group_id, chat_type, role, content, sender_name, qq_message_id, embedding, token_count, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) ON CONFLICT (message_uuid) DO NOTHING",
                )
                .bind(message_uuid).bind(user_id).bind(group_id).bind(chat_type).bind(role)
                .bind(content).bind(sender_name).bind(qq_message_id).bind(embedding_vec)
                .bind(token_count).bind(created_at).execute(&self.pool).await?;

            inserted += result.rows_affected() as usize;
        }
        Ok(inserted)
    }

    pub async fn cleanup_expired_memories(&self) -> Result<u64> {
        let result = sqlx::query("DELETE FROM dialogues WHERE expires_at IS NOT NULL AND expires_at < NOW()")
            .execute(&self.pool).await?;

        let count = result.rows_affected();
        if count > 0 { log::info!("🗑️  清理了 {} 条过期记忆", count); }
        Ok(count)
    }

    async fn try_create_vector_indexes(&self) -> Result<()> {
        let mut success = true;
        
        let group_index_sql = format!(
            r#"CREATE INDEX IF NOT EXISTS idx_group_embedding ON dialogues 
            USING ivfflat (embedding vector_cosine_ops) WITH (lists={})
            WHERE group_id IS NOT NULL"#, self.vector_config.lists
        );
        
        match sqlx::query(&group_index_sql).execute(&self.pool).await {
            Ok(_) => log::info!("   ✓ 群聊向量索引创建成功 (lists={})", self.vector_config.lists),
            Err(e) => { log::warn!("   ⚠ 群聊向量索引创建失败: {}", e); success = false; }
        }

        let private_index_sql = format!(
            r#"CREATE INDEX IF NOT EXISTS idx_private_embedding ON dialogues 
            USING ivfflat (embedding vector_cosine_ops) WITH (lists={})
            WHERE group_id IS NULL"#, self.vector_config.lists
        );
        
        match sqlx::query(&private_index_sql).execute(&self.pool).await {
            Ok(_) => log::info!("   ✓ 私聊向量索引创建成功 (lists={})", self.vector_config.lists),
            Err(e) => { log::warn!("   ⚠ 私聊向量索引创建失败: {}", e); success = false; }
        }

        if success {
            self.vector_indexes_created.store(true, Ordering::Relaxed);
            log::info!("✅ 向量索引自动创建完成");
        }
        Ok(())
    }
}

