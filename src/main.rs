use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

struct AppState {
    model: Mutex<TextEmbedding>,
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    #[serde(default = "default_model")]
    model: String,
    input: InputData,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum InputData {
    One(String),
    Many(Vec<String>),
}

fn default_model() -> String {
    "bge-small-zh-v1.5".to_string()
}

#[derive(Serialize)]
struct EmbeddingItem {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct EmbeddingResponse {
    object: &'static str,
    data: Vec<EmbeddingItem>,
    model: String,
    usage: Usage,
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "device": "cpu",
        "model": "BAAI/bge-small-zh-v1.5"
    }))
}

async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, String)> {
    let texts = match req.input {
        InputData::One(text) => vec![text],
        InputData::Many(texts) => texts,
    };

    if texts.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "input must not be empty".into()));
    }

    let mut model = state.model.lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model lock poisoned".into()))?;

    let vectors = model.embed(texts, None)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let data = vectors.into_iter().enumerate().map(|(index, embedding)| {
        EmbeddingItem {
            object: "embedding",
            index,
            embedding,
        }
    }).collect();

    Ok(Json(EmbeddingResponse {
        object: "list",
        data,
        model: req.model,
        // Token counting can be added later.
        usage: Usage { prompt_tokens: 0, total_tokens: 0 },
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cache_dir = std::env::var("FASTEMBED_CACHE_DIR")
        .unwrap_or_else(|_| format!("{}/.cache/fastembed", std::env::var("HOME").unwrap()));

    std::fs::create_dir_all(&cache_dir)?;

    println!("Loading BAAI/bge-small-zh-v1.5 on CPU...");

    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallZHV15)
            .with_cache_dir(cache_dir.into())
    )?;

    let state = Arc::new(AppState { model: Mutex::new(model) });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/embeddings", post(embeddings))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8765").await?;

    println!("Embedding service listening on http://127.0.0.1:8765");
    axum::serve(listener, app).await?;

    Ok(())
}
