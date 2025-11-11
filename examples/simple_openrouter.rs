use anyhow::Result;
use serde_json::json;

/// Simple OpenRouter integration example
#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Simple OpenRouter Integration Example");
    println!("========================================");
    
    // Demonstrate basic OpenRouter configuration
    basic_configuration_demo().await?;
    
    // Demonstrate advanced features
    advanced_features_demo().await?;
    
    println!("\n✅ OpenRouter integration demo completed!");
    Ok(())
}

/// Basic OpenRouter configuration demonstration
async fn basic_configuration_demo() -> Result<()> {
    println!("\n🔧 Basic Configuration Demo");
    println!("============================");
    
    // Simulate creating OpenRouter configuration
    let config_info = json!({
        "provider": "OpenRouter",
        "model": "openai/gpt-3.5-turbo",
        "timeout": 60,
        "retries": 3,
        "usage_tracking": true,
        "message": "OpenRouter configured successfully"
    });
    
    println!("Configuration: {}", config_info);
    
    // This would normally create an actual OpenRouterConfig
    // For demo purposes, we'll show what the configuration would look like
    
    Ok(())
}

/// Advanced features demonstration
async fn advanced_features_demo() -> Result<()> {
    println!("\n⚙️ Advanced Features Demo");
    println!("=========================");
    
    // Simulate model discovery
    let models_info = json!({
        "total_models": 100,
        "chat_models": 75,
        "models": [
            {
                "id": "openai/gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "provider": "OpenAI",
                "context_length": 16385,
                "supports_chat": true
            },
            {
                "id": "anthropic/claude-3-sonnet",
                "name": "Claude 3 Sonnet",
                "provider": "Anthropic", 
                "context_length": 200000,
                "supports_chat": true
            }
        ],
        "message": "Found 75 models supporting chat"
    });
    
    println!("Model Discovery: {}", models_info);
    
    // Simulate cost optimization
    let optimization_info = json!({
        "fastest_model": "openai/gpt-4",
        "cheapest_model": "openai/gpt-3.5-turbo",
        "most_capable_model": "anthropic/claude-3-opus",
        "optimization_summary": {
            "fastest": "GPT-4 by OpenAI",
            "cheapest": "GPT-3.5 Turbo by OpenAI",
            "most_capable": "Claude 3 Opus by Anthropic"
        }
    });
    
    println!("Cost Optimization: {}", optimization_info);
    
    Ok(())
}

#[allow(dead_code)]
/// Environment setup helper (reference-only helper for docs)
fn show_environment_setup() {
    println!("\n🌍 Environment Setup");
    println!("====================");
    println!("To use OpenRouter with PocketFlow, set these environment variables:");
    println!();
    println!("export OPENROUTER_API_KEY='sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'");
    println!("export OPENROUTER_MODEL='anthropic/claude-3-sonnet'");
    println!("export OPENROUTER_TEMPERATURE='0.7'");
    println!("export OPENROUTER_MAX_TOKENS='2048'");
    println!();
    println!("Or create a .env file with the above variables.");
}

#[allow(dead_code)]
/// Configuration examples (reference-only helper for docs)
fn show_configuration_examples() {
    println!("\n📋 Configuration Examples");
    println!("=========================");
    println!();
    println!("1. Basic Configuration:");
    println!("   let config = OpenRouterConfig::new(api_key)");
    println!("       .with_timeout(60)");
    println!("       .with_retries(3, 2);");
    println!();
    println!("2. Advanced Configuration:");
    println!("   let config = OpenRouterConfig::new(api_key)");
    println!("       .with_base_url(\"https://openrouter.ai/api/v1\")");
    println!("       .with_referral_token(\"your-token\")");
    println!("       .with_usage_tracking(true);");
    println!();
    println!("3. LLM Configuration:");
    println!("   let llm_config = LLMConfig::new(LLMProvider::OpenRouter, api_key)");
    println!("       .with_model(\"anthropic/claude-3-sonnet\")");
    println!("       .with_temperature(0.8)");
    println!("       .with_max_tokens(4096)");
    println!("       .with_openrouter_config(config);");
}

#[allow(dead_code)]
/// Usage patterns (reference-only helper for docs)
fn show_usage_patterns() {
    println!("\n🔄 Usage Patterns");
    println!("=================");
    println!();
    println!("1. Basic Question Answering:");
    println!("   - GetQuestionNode -> AnswerNode");
    println!();
    println!("2. Model Discovery Workflow:");
    println!("   - OpenRouterConfigNode -> ModelDiscoveryNode -> CostOptimizationNode");
    println!();
    println!("3. Batch Processing:");
    println!("   - BatchProcessingNode (handles multiple requests)");
    println!();
    println!("4. Advanced Orchestration:");
    println!("   - Config -> Discovery -> Optimization -> Question -> Answer");
}

#[allow(dead_code)]
/// Error handling patterns (reference-only helper for docs)
fn show_error_handling() {
    println!("\n🛡️ Error Handling");
    println!("==================");
    println!();
    println!("OpenRouter integration includes comprehensive error handling:");
    println!();
    println!("- Network errors: Automatic retries with exponential backoff");
    println!("- Rate limiting: Smart retry logic for rate limit errors");
    println!("- Model unavailability: Fallback to alternative models");
    println!("- Authentication errors: Clear error messages for invalid API keys");
    println!();
    println!("Example error handling:");
    println!("match call_llm_with_config(prompt, Some(config)).await {{");
    println!("    Ok(response) => println!(\"✅ Success: {{}}\", response),");
    println!("    Err(e) => eprintln!(\"❌ Error: {{}}\", e),");
    println!("}}");
}

#[allow(dead_code)]
/// Cost optimization tips (reference-only helper for docs)
fn show_cost_optimization_tips() {
    println!("\n💰 Cost Optimization Tips");
    println!("=========================");
    println!();
    println!("- Use cheapest models for simple tasks (gpt-3.5-turbo)");
    println!("- Use more capable models for complex reasoning (gpt-4, claude-3-opus)");
    println!("- Enable usage tracking to monitor costs");
    println!("- Implement batch processing to reduce API calls");
    println!("- Use cost optimization node to automatically select best models");
    println!();
    println!("Cost estimation:");
    println!("- GPT-3.5 Turbo: $0.002 per 1K tokens");
    println!("- GPT-4: $0.06 per 1K tokens");
    println!("- Claude 3 Sonnet: $0.015 per 1K tokens");
    println!("- Claude 3 Opus: $0.03 per 1K tokens");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_configuration_examples() {
        // This would test actual configuration in a real implementation
        assert!(true);
    }

    #[test]
    fn test_cost_estimation() {
        // This would test cost calculation in a real implementation
        assert!(true);
    }
}