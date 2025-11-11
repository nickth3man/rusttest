mod nodes;
mod state;
mod utils;

use anyhow::Result;
use nodes::{
    AnswerNode, BatchProcessingNode, CostOptimizationNode, GetQuestionNode, ModelDiscoveryNode,
    OpenRouterConfigNode,
};
use pocketflow_rs::{build_flow, Context};
use serde_json::json;
use state::MyState;
use utils::{LLMConfig, LLMProvider, OpenRouterConfig};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 PocketFlow with OpenRouter Integration Demo");
    println!("=============================================");

    // Check for API key in environment
    let openrouter_key =
        std::env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| "dummy-key".to_string());

    if openrouter_key == "dummy-key" {
        println!("⚠️  No OpenRouter API key found. Using mock responses.");
        println!("   Set OPENROUTER_API_KEY environment variable for real API calls.");
    }

    // Example 1: Basic OpenRouter Flow
    println!("\n📝 Example 1: Basic OpenRouter Question Answering");
    basic_openrouter_flow(&openrouter_key).await?;

    // Example 2: Advanced OpenRouter Configuration Flow
    println!("\n⚙️  Example 2: Advanced OpenRouter Configuration");
    advanced_openrouter_flow(&openrouter_key).await?;

    // Example 3: Model Discovery and Selection Flow
    println!("\n🔍 Example 3: Model Discovery and Cost Optimization");
    model_discovery_flow(&openrouter_key).await?;

    // Example 4: Batch Processing Flow
    println!("\n📦 Example 4: Batch Processing with OpenRouter");
    batch_processing_flow(&openrouter_key).await?;

    println!("\n✅ All OpenRouter examples completed successfully!");
    Ok(())
}

/// Basic OpenRouter flow with simple question answering
async fn basic_openrouter_flow(api_key: &str) -> Result<()> {
    // Create OpenRouter configuration
    let openrouter_config = OpenRouterConfig::new(api_key)
        .with_timeout(60)
        .with_retries(3, 2)
        .with_usage_tracking(true);

    let llm_config = LLMConfig::new(LLMProvider::OpenRouter, api_key.to_string())
        .with_openrouter_config(openrouter_config);

    // Instantiate nodes
    let get_question = GetQuestionNode;
    let answer = AnswerNode;

    // Build flow with context sharing configuration
    let flow = build_flow!(
        start: ("get_question", get_question),
        nodes: [("answer", answer)],
        edges: [
            ("get_question", "answer", MyState::Success)
        ]
    );

    // Shared context with LLM configuration
    let mut context = Context::new();
    context.set("llm_config", serde_json::to_value(llm_config)?);

    // Run flow
    let result_context = flow.run(context).await?;

    // Display results
    if let Some(q) = result_context.get("question").and_then(|v| v.as_str()) {
        println!("   Question: {}", q);
    }
    if let Some(a) = result_context.get("answer").and_then(|v| v.as_str()) {
        println!("   Answer: {}", a);
    }

    Ok(())
}

/// Advanced OpenRouter configuration with custom settings
async fn advanced_openrouter_flow(api_key: &str) -> Result<()> {
    // Create advanced OpenRouter configuration
    let openrouter_config = OpenRouterConfig::new(api_key)
        .with_base_url("https://openrouter.ai/api/v1")
        .with_referral_token("your-referral-token")
        .with_header("Custom-Header", "custom-value")
        .with_timeout(120)
        .with_retries(5, 3);

    let llm_config = LLMConfig::new(LLMProvider::OpenRouter, api_key.to_string())
        .with_model("anthropic/claude-3-sonnet")
        .with_temperature(0.7)
        .with_max_tokens(2048)
        .with_openrouter_config(openrouter_config);

    // Build advanced flow with configuration node
    let openrouter_config_node = OpenRouterConfigNode;
    let get_question = GetQuestionNode;
    let answer = AnswerNode;

    let flow = build_flow!(
        start: ("config", openrouter_config_node),
        nodes: [
            ("get_question", get_question),
            ("answer", answer)
        ],
        edges: [
            ("config", "get_question", MyState::Success),
            ("get_question", "answer", MyState::Success)
        ]
    );

    let mut context = Context::new();
    context.set("llm_config", serde_json::to_value(llm_config)?);
    let result_context = flow.run(context).await?;

    if let Some(config_info) = result_context.get("config_info").and_then(|v| v.as_str()) {
        println!("   Configuration: {}", config_info);
    }

    Ok(())
}

/// Model discovery and cost optimization flow
async fn model_discovery_flow(api_key: &str) -> Result<()> {
    let model_discovery = ModelDiscoveryNode;
    let cost_optimizer = CostOptimizationNode;
    let get_question = GetQuestionNode;
    let answer = AnswerNode;

    let flow = build_flow!(
        start: ("discover_models", model_discovery),
        nodes: [
            ("optimize_cost", cost_optimizer),
            ("get_question", get_question),
            ("answer", answer)
        ],
        edges: [
            ("discover_models", "optimize_cost", MyState::Success),
            ("optimize_cost", "get_question", MyState::Success),
            ("get_question", "answer", MyState::Success)
        ]
    );

    let mut context = Context::new();
    context.set("api_key", json!(api_key));

    let result_context = flow.run(context).await?;

    if let Some(best_model) = result_context.get("best_model").and_then(|v| v.as_str()) {
        println!("   Best model for your use case: {}", best_model);
    }

    Ok(())
}

/// Batch processing flow with multiple questions
async fn batch_processing_flow(api_key: &str) -> Result<()> {
    let batch_processor = BatchProcessingNode;
    let get_question = GetQuestionNode;
    let answer = AnswerNode;

    let flow = build_flow!(
        start: ("batch_process", batch_processor),
        nodes: [
            ("get_question", get_question),
            ("answer", answer)
        ],
        edges: [
            ("batch_process", "get_question", MyState::Success),
            ("get_question", "answer", MyState::Success)
        ]
    );

    let mut context = Context::new();
    context.set("api_key", json!(api_key));

    let result_context = flow.run(context).await?;

    if let Some(batch_results) = result_context.get("batch_results").and_then(|v| v.as_str()) {
        println!("   Batch processing completed: {}", batch_results);
    }

    Ok(())
}
