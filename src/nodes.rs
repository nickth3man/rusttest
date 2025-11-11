use crate::state::MyState;
use crate::utils::{
    call_llm, call_llm_with_config, get_best_model, list_openrouter_models, LLMConfig, LLMProvider,
    ModelSelectionCriteria, OpenRouterConfig,
};
use anyhow::Result;
use async_trait::async_trait;
use pocketflow_rs::{Context, Node, ProcessResult};
use serde_json::{json, Value};
use tokio::io::{self, AsyncBufReadExt, AsyncReadExt};

pub struct GetQuestionNode;

#[async_trait]
impl Node for GetQuestionNode {
    type State = MyState;

    async fn execute(&self, _context: &Context) -> Result<serde_json::Value> {
        // Check if there's piped input available
        let mut question = String::new();

        // Try to read from stdin with a timeout
        match tokio::time::timeout(
            std::time::Duration::from_millis(100),
            tokio::io::stdin().read_to_string(&mut question),
        )
        .await
        {
            Ok(Ok(_)) => {
                // We got piped input
                question = question.trim().to_string();
            }
            _ => {
                // No piped input, prompt for interactive input
                println!("Enter your question: ");
                let mut reader = io::BufReader::new(tokio::io::stdin());
                let mut line = String::new();
                reader.read_line(&mut line).await?;
                question = line.trim().to_string();
            }
        }

        Ok(json!(question))
    }

    async fn post_process(
        &self,
        context: &mut Context,
        result: &Result<serde_json::Value>,
    ) -> Result<ProcessResult<MyState>> {
        if let Ok(val) = result {
            context.set("question", val.clone());
            Ok(ProcessResult::new(MyState::Success, "success".to_string()))
        } else {
            Ok(ProcessResult::new(MyState::Failure, "failure".to_string()))
        }
    }
}

// ================================
// OpenRouter-Specific Nodes
// ================================

/// Node for configuring OpenRouter settings
pub struct OpenRouterConfigNode;

#[async_trait]
impl Node for OpenRouterConfigNode {
    type State = MyState;

    async fn execute(&self, context: &Context) -> Result<serde_json::Value> {
        let api_key = context
            .get("api_key")
            .and_then(|v| v.as_str())
            .unwrap_or("dummy-key");

        // Create OpenRouter configuration
        let config = OpenRouterConfig::new(api_key)
            .with_timeout(60)
            .with_retries(3, 2)
            .with_usage_tracking(true)
            .with_header("User-Agent", "PocketFlow-OpenRouter-Demo");

        // Create LLM configuration
        let llm_config = LLMConfig::new(LLMProvider::OpenRouter, api_key.to_string())
            .with_model("anthropic/claude-3-sonnet")
            .with_temperature(0.7)
            .with_max_tokens(2048)
            .with_openrouter_config(config);

        Ok(json!({
            "config_info": format!("OpenRouter configured with model: {}", llm_config.model),
            "model": llm_config.model,
            "provider": "OpenRouter",
            "timeout": 60,
            "retries": 3,
            "usage_tracking": true
        }))
    }

    async fn post_process(
        &self,
        context: &mut Context,
        result: &Result<serde_json::Value>,
    ) -> Result<ProcessResult<MyState>> {
        if let Ok(val) = result {
            context.set("config_info", val.clone());
            Ok(ProcessResult::new(MyState::Success, "success".to_string()))
        } else {
            Ok(ProcessResult::new(MyState::Failure, "failure".to_string()))
        }
    }
}

/// Node for discovering available OpenRouter models
pub struct ModelDiscoveryNode;

#[async_trait]
impl Node for ModelDiscoveryNode {
    type State = MyState;

    async fn execute(&self, context: &Context) -> Result<serde_json::Value> {
        let api_key = context
            .get("api_key")
            .and_then(|v| v.as_str())
            .unwrap_or("dummy-key");

        let config = OpenRouterConfig::new(api_key);
        let models = list_openrouter_models(&config).await?;

        // Get model information
        let model_info: Vec<Value> = models
            .iter()
            .filter(|m| m.supports_chat)
            .take(10) // Show top 10 models
            .map(|m| {
                json!({
                    "id": m.id,
                    "name": m.name,
                    "provider": m.provider,
                    "context_length": m.context_length,
                    "supports_chat": m.supports_chat,
                    "supports_completion": m.supports_completion
                })
            })
            .collect();

        Ok(json!({
            "total_models": models.len(),
            "chat_models": model_info.len(),
            "models": model_info,
            "message": format!("Found {} models supporting chat", model_info.len())
        }))
    }

    async fn post_process(
        &self,
        context: &mut Context,
        result: &Result<serde_json::Value>,
    ) -> Result<ProcessResult<MyState>> {
        if let Ok(val) = result {
            context.set("model_discovery", val.clone());
            Ok(ProcessResult::new(MyState::Success, "success".to_string()))
        } else {
            Ok(ProcessResult::new(MyState::Failure, "failure".to_string()))
        }
    }
}

/// Node for cost optimization and model selection
pub struct CostOptimizationNode;

#[async_trait]
impl Node for CostOptimizationNode {
    type State = MyState;

    async fn execute(&self, context: &Context) -> Result<serde_json::Value> {
        let api_key = context
            .get("api_key")
            .and_then(|v| v.as_str())
            .unwrap_or("dummy-key");

        let config = OpenRouterConfig::new(api_key);
        let models = list_openrouter_models(&config).await?;

        // Find best models based on different criteria
        let fastest = get_best_model(&models, ModelSelectionCriteria::Fastest);
        let cheapest = get_best_model(&models, ModelSelectionCriteria::Cheapest);
        let most_capable = get_best_model(&models, ModelSelectionCriteria::MostCapable);
        let openai_model = get_best_model(
            &models,
            ModelSelectionCriteria::ByProvider("OpenAI".to_string()),
        );

        Ok(json!({
            "fastest_model": fastest.map(|m| m.id.clone()),
            "cheapest_model": cheapest.map(|m| m.id.clone()),
            "most_capable_model": most_capable.map(|m| m.id.clone()),
            "openai_model": openai_model.map(|m| m.id.clone()),
            "optimization_summary": {
                "fastest": fastest.map(|m| format!("{} by {}", m.name, m.provider)),
                "cheapest": cheapest.map(|m| format!("{} by {}", m.name, m.provider)),
                "most_capable": most_capable.map(|m| format!("{} by {}", m.name, m.provider)),
                "openai_recommended": openai_model.map(|m| format!("{} by {}", m.name, m.provider))
            }
        }))
    }

    async fn post_process(
        &self,
        context: &mut Context,
        result: &Result<serde_json::Value>,
    ) -> Result<ProcessResult<MyState>> {
        if let Ok(val) = result {
            context.set("optimization_result", val.clone());

            // Set the best model for subsequent nodes
            if let Some(best_model) = val.get("cheapest_model").and_then(|m| m.as_str()) {
                context.set("best_model", json!(best_model));
            }

            Ok(ProcessResult::new(MyState::Success, "success".to_string()))
        } else {
            Ok(ProcessResult::new(MyState::Failure, "failure".to_string()))
        }
    }
}

/// Node for batch processing multiple requests
pub struct BatchProcessingNode;

#[async_trait]
impl Node for BatchProcessingNode {
    type State = MyState;

    async fn execute(&self, context: &Context) -> Result<serde_json::Value> {
        let api_key = context
            .get("api_key")
            .and_then(|v| v.as_str())
            .unwrap_or("dummy-key");

        // Sample batch questions
        let batch_questions = vec![
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "Write a haiku about programming",
            "What are the benefits of using OpenRouter?",
            "Compare GPT-4 and Claude 3",
        ];

        let mut results = Vec::new();

        for (i, question) in batch_questions.iter().enumerate() {
            let llm_config = LLMConfig::new(LLMProvider::OpenRouter, api_key.to_string())
                .with_model("openai/gpt-3.5-turbo")
                .with_temperature(0.7);

            match call_llm_with_config(question, Some(llm_config)).await {
                Ok(response) => {
                    results.push(json!({
                        "question": question,
                        "answer": response,
                        "status": "success",
                        "question_number": i + 1
                    }));
                }
                Err(e) => {
                    results.push(json!({
                        "question": question,
                        "error": e.to_string(),
                        "status": "failed",
                        "question_number": i + 1
                    }));
                }
            }
        }

        let successful = results
            .iter()
            .filter(|r| r.get("status") == Some(&json!("success")))
            .count();
        let failed = results.len() - successful;

        Ok(json!({
            "batch_size": results.len(),
            "successful": successful,
            "failed": failed,
            "results": results,
            "summary": format!("Processed {} questions: {} successful, {} failed",
                              results.len(), successful, failed)
        }))
    }

    async fn post_process(
        &self,
        context: &mut Context,
        result: &Result<serde_json::Value>,
    ) -> Result<ProcessResult<MyState>> {
        if let Ok(val) = result {
            context.set("batch_results", val.clone());
            Ok(ProcessResult::new(MyState::Success, "success".to_string()))
        } else {
            Ok(ProcessResult::new(MyState::Failure, "failure".to_string()))
        }
    }
}

#[allow(dead_code)]
/// Advanced node for usage tracking and analytics
/// Kept as an example node for future usage analytics flows.
pub struct UsageTrackingNode;

#[async_trait]
impl Node for UsageTrackingNode {
    type State = MyState;

    async fn execute(&self, _context: &Context) -> Result<serde_json::Value> {
        // Simulate usage tracking
        let mock_usage = json!({
            "total_requests": 42,
            "total_tokens": 15678,
            "total_cost": 2.34,
            "average_tokens_per_request": 373,
            "most_used_model": "openai/gpt-3.5-turbo",
            "models_used": ["openai/gpt-3.5-turbo", "anthropic/claude-3-sonnet", "openai/gpt-4"],
            "cost_breakdown": {
                "openai/gpt-3.5-turbo": 1.23,
                "anthropic/claude-3-sonnet": 0.78,
                "openai/gpt-4": 0.33
            },
            "usage_patterns": {
                "peak_hour": "14:00",
                "peak_day": "Wednesday",
                "average_response_time": "2.3s"
            }
        });

        Ok(mock_usage)
    }

    async fn post_process(
        &self,
        context: &mut Context,
        result: &Result<serde_json::Value>,
    ) -> Result<ProcessResult<MyState>> {
        if let Ok(val) = result {
            context.set("usage_analytics", val.clone());
            Ok(ProcessResult::new(MyState::Success, "success".to_string()))
        } else {
            Ok(ProcessResult::new(MyState::Failure, "failure".to_string()))
        }
    }
}

pub struct AnswerNode;

#[async_trait]
impl Node for AnswerNode {
    type State = MyState;

    async fn execute(&self, context: &Context) -> Result<serde_json::Value> {
        let question = context
            .get("question")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let answer = call_llm(&question).await?;
        Ok(json!(answer))
    }

    async fn post_process(
        &self,
        context: &mut Context,
        result: &Result<serde_json::Value>,
    ) -> Result<ProcessResult<MyState>> {
        if let Ok(val) = result {
            context.set("answer", val.clone());
            Ok(ProcessResult::new(MyState::Success, "success".to_string()))
        } else {
            Ok(ProcessResult::new(MyState::Failure, "failure".to_string()))
        }
    }
}
