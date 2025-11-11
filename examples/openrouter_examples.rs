use anyhow::Result;
use pocketflow_rs::{build_flow, Context};
use serde_json::json;

// Re-define the types needed for this example since it's a standalone example
#[derive(Debug, Clone, PartialEq)]
pub enum MyState {
    Success,
    Failure,
    Default,
}

impl pocketflow_rs::ProcessState for MyState {
    fn is_default(&self) -> bool {
        matches!(self, MyState::Default)
    }
    
    fn to_condition(&self) -> String {
        match self {
            MyState::Success => "success".to_string(),
            MyState::Failure => "failure".to_string(),
            MyState::Default => "default".to_string(),
        }
    }
}

impl Default for MyState {
    fn default() -> Self {
        MyState::Default
    }
}

// Simple node implementations for demonstration
use async_trait::async_trait;

pub struct GetQuestionNode;

#[async_trait]
impl pocketflow_rs::Node for GetQuestionNode {
    type State = MyState;

    async fn execute(&self, _context: &Context) -> Result<serde_json::Value> {
        Ok(json!("What is the capital of France?"))
    }

    async fn post_process(
        &self,
        context: &mut Context,
        result: &Result<serde_json::Value>,
    ) -> Result<pocketflow_rs::ProcessResult<MyState>> {
        if let Ok(val) = result {
            context.set("question", val.clone());
            Ok(pocketflow_rs::ProcessResult::new(MyState::Success, "success".to_string()))
        } else {
            Ok(pocketflow_rs::ProcessResult::new(MyState::Failure, "failure".to_string()))
        }
    }
}

pub struct AnswerNode;

#[async_trait]
impl pocketflow_rs::Node for AnswerNode {
    type State = MyState;

    async fn execute(&self, context: &Context) -> Result<serde_json::Value> {
        let question = context
            .get("question")
            .and_then(|v| v.as_str())
            .unwrap_or("No question provided");
        
        // Simulate LLM response
        let answer = format!("The answer to '{}' is Paris.", question);
        Ok(json!(answer))
    }

    async fn post_process(
        &self,
        context: &mut Context,
        result: &Result<serde_json::Value>,
    ) -> Result<pocketflow_rs::ProcessResult<MyState>> {
        if let Ok(val) = result {
            context.set("answer", val.clone());
            Ok(pocketflow_rs::ProcessResult::new(MyState::Success, "success".to_string()))
        } else {
            Ok(pocketflow_rs::ProcessResult::new(MyState::Failure, "failure".to_string()))
        }
    }
}

// Configuration types for demonstration
#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
}

impl OpenRouterConfig {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            timeout_seconds: 60,
            max_retries: 3,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub enum LLMProvider {
    OpenRouter,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LLMConfig {
    pub provider: LLMProvider,
    pub api_key: String,
    pub model: String,
}

impl LLMConfig {
    pub fn new(provider: LLMProvider, api_key: String) -> Self {
        Self {
            provider,
            api_key,
            model: "openai/gpt-3.5-turbo".to_string(),
        }
    }
    
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }
}

/// Example demonstrating OpenRouter integration with PocketFlow
#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 PocketFlow OpenRouter Examples");
    println!("==================================");
    
    // Example 1: Basic Configuration
    basic_configuration_example().await?;
    
    // Example 2: Advanced Configuration
    advanced_configuration_example().await?;
    
    // Example 3: Model Discovery
    model_discovery_example().await?;
    
    // Example 4: Batch Processing
    batch_processing_example().await?;
    
    println!("\n✅ All examples completed!");
    Ok(())
}

/// Basic OpenRouter configuration example
async fn basic_configuration_example() -> Result<()> {
    println!("\n🔧 Basic Configuration Example");
    println!("===============================");
    
    // Create basic OpenRouter configuration
    let _openrouter_config = OpenRouterConfig::new("dummy-key"); // underscore: used for illustrative config
    let _llm_config = LLMConfig::new(LLMProvider::OpenRouter, "dummy-key".to_string()); // underscore: used for illustrative config

    // Create nodes
    let get_question = GetQuestionNode;
    let answer = AnswerNode;

    // Build flow
    let flow = build_flow!(
        start: ("get_question", get_question),
        nodes: [("answer", answer)],
        edges: [
            ("get_question", "answer", MyState::Success)
        ]
    );

    // Context with configuration (store illustrative LLM config)
    let mut context = Context::new();
    context.set("llm_config", serde_json::to_value(_llm_config)?);

    let result = flow.run(context).await?;
    
    if let Some(response) = result.get("answer").and_then(|v| v.as_str()) {
        println!("Response: {}", response);
    }

    Ok(())
}

/// Advanced OpenRouter configuration with custom settings
async fn advanced_configuration_example() -> Result<()> {
    println!("\n⚙️ Advanced Configuration Example");
    println!("==================================");
    
    // Create advanced configuration
    let _openrouter_config = OpenRouterConfig::new("dummy-key"); // underscore: used for illustrative config
    let _llm_config = LLMConfig::new(LLMProvider::OpenRouter, "dummy-key".to_string()); // underscore: used for illustrative config

    // Use configuration node
    let config_node = GetQuestionNode; // Using GetQuestionNode as placeholder
    let get_question = GetQuestionNode;
    let answer = AnswerNode;

    let flow = build_flow!(
        start: ("config", config_node),
        nodes: [
            ("get_question", get_question),
            ("answer", answer)
        ],
        edges: [
            ("config", "get_question", MyState::Success),
            ("get_question", "answer", MyState::Success)
        ]
    );

    let context = Context::new();
    let result = flow.run(context).await?;
    
    if let Some(config_info) = result.get("config_info").and_then(|v| v.as_str()) {
        println!("Configuration: {}", config_info);
    }

    Ok(())
}

/// Model discovery and selection example
async fn model_discovery_example() -> Result<()> {
    println!("\n🔍 Model Discovery Example");
    println!("==========================");
    
    let model_discovery = GetQuestionNode; // Using GetQuestionNode as placeholder
    let cost_optimizer = AnswerNode; // Using AnswerNode as placeholder

    let flow = build_flow!(
        start: ("discover", model_discovery),
        nodes: [("optimize", cost_optimizer)],
        edges: [
            ("discover", "optimize", MyState::Success)
        ]
    );

    let mut context = Context::new();
    context.set("api_key", json!("dummy-key"));
    
    let result = flow.run(context).await?;
    
    if let Some(model_discovery) = result.get("model_discovery") {
        println!("Model Discovery Result: {}", model_discovery);
    }
    
    if let Some(optimization) = result.get("optimization_result") {
        println!("Optimization Result: {}", optimization);
    }

    Ok(())
}

/// Batch processing example
async fn batch_processing_example() -> Result<()> {
    println!("\n📦 Batch Processing Example");
    println!("==========================");
    
    let batch_processor = GetQuestionNode; // Using GetQuestionNode as placeholder

    let flow = build_flow!(
        start: ("batch", batch_processor),
        nodes: [],
        edges: []
    );

    let mut context = Context::new();
    context.set("api_key", json!("dummy-key"));
    
    let result = flow.run(context).await?;
    
    if let Some(batch_results) = result.get("batch_results") {
        println!("Batch Results: {}", batch_results);
    }

    Ok(())
}

#[allow(dead_code)]
/// Environment configuration helper (reference-only helper for docs)
fn print_environment_guide() {
    println!("\n🌍 Environment Configuration Guide");
    println!("===================================");
    println!("Set these environment variables for OpenRouter:");
    println!("  OPENROUTER_API_KEY=your-api-key-here");
    println!("  OPENROUTER_MODEL=openai/gpt-3.5-turbo");
    println!("  OPENROUTER_TEMPERATURE=0.7");
    println!("  OPENROUTER_MAX_TOKENS=1024");
    println!("  OPENROUTER_TIMEOUT=60");
    println!("  OPENROUTER_RETRIES=3");
    println!();
    println!("Example .env file content:");
    println!("OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
    println!("OPENROUTER_MODEL=anthropic/claude-3-sonnet");
    println!("OPENROUTER_TEMPERATURE=0.8");
    println!("OPENROUTER_MAX_TOKENS=2048");
}

#[allow(dead_code)]
/// Configuration validation helper (reference-only helper for docs)
fn validate_openrouter_config(config: &OpenRouterConfig) -> Result<()> {
    if config.api_key.is_empty() || config.api_key == "your-api-key" {
        anyhow::bail!("OpenRouter API key is required and should not be the placeholder value");
    }
    
    if !config.base_url.starts_with("https://") {
        anyhow::bail!("OpenRouter base URL should be a valid HTTPS URL");
    }
    
    if config.timeout_seconds == 0 {
        anyhow::bail!("Timeout should be greater than 0");
    }
    
    if config.max_retries == 0 {
        anyhow::bail!("Max retries should be greater than 0");
    }
    
    Ok(())
}

#[allow(dead_code)]
/// Cost estimation helper (reference-only helper for docs)
fn estimate_cost(model: &str, prompt_tokens: u32, completion_tokens: u32) -> f64 {
    let cost_per_1k_tokens = match model {
        "gpt-4" | "gpt-4-32k" => 0.06,
        "gpt-3.5-turbo" => 0.002,
        "claude-3-sonnet" => 0.015,
        "claude-3-opus" => 0.03,
        _ => 0.01,
    };
    
    let total_tokens = prompt_tokens + completion_tokens;
    (total_tokens as f64 / 1000.0) * cost_per_1k_tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_config() {
        // Use the local OpenRouterConfig defined in this example module
        let config = OpenRouterConfig::new("test-key");
        assert!(validate_openrouter_config(&config).is_ok());
        
        let invalid_config = OpenRouterConfig::new("");
        assert!(validate_openrouter_config(&invalid_config).is_err());
    }

    #[test]
    fn test_cost_estimation() {
        let cost = estimate_cost("gpt-3.5-turbo", 100, 200);
        assert_eq!(cost, 0.0006); // 300 tokens * $0.002/1000
        
        let cost = estimate_cost("gpt-4", 500, 500);
        assert_eq!(cost, 0.06); // 1000 tokens * $0.06/1000
    }
}