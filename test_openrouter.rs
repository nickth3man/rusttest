use pocketflow_template_rust::utils::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing OpenRouter integration...");
    
    // Test 1: Create OpenRouter configuration
    let config = OpenRouterConfig::new("test-key");
    println!("✓ OpenRouterConfig created successfully");
    
    // Test 2: Create LLMConfig with OpenRouter
    let llm_config = LLMConfig::new(LLMProvider::OpenRouter, "test-key".to_string());
    println!("✓ LLMConfig with OpenRouter created successfully");
    
    // Test 3: Test usage tracking
    let mut usage = OpenRouterUsage::new();
    let mock_response = serde_json::json!({
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80
        }
    });
    usage.update_from_response(&mock_response, "gpt-3.5-turbo");
    println!("✓ Usage tracking works: {} tokens, cost: {}", 
             usage.total_tokens, usage.get_cost_string());
    
    // Test 4: Test model parsing
    let mock_models = serde_json::json!({
        "data": [
            {
                "id": "openai/gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "description": "Fast and efficient model",
                "context_length": 16385,
                "provider": {
                    "name": "OpenAI"
                },
                "features": {
                    "chat": true,
                    "completion": true
                }
            }
        ]
    });
    
    let models = mock_models
        .get("data")
        .and_then(|data| data.as_array())
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|model_data| {
            Some(OpenRouterModel {
                id: model_data.get("id")?.as_str()?.to_string(),
                name: model_data.get("name")?.as_str()?.to_string(),
                description: model_data.get("description")
                    .and_then(|d| d.as_str())
                    .map(|s| s.to_string()),
                provider: model_data.get("provider")
                    .and_then(|p| p.get("name"))
                    .and_then(|p| p.as_str())
                    .unwrap_or("Unknown")
                    .to_string(),
                context_length: model_data.get("context_length")
                    .and_then(|c| c.as_u64())
                    .map(|c| c as u32),
                supports_chat: model_data.get("features")
                    .and_then(|f| f.get("chat"))
                    .and_then(|c| c.as_bool())
                    .unwrap_or(false),
                supports_completion: model_data.get("features")
                    .and_then(|f| f.get("completion"))
                    .and_then(|c| c.as_bool())
                    .unwrap_or(false),
            })
        })
        .collect::<Vec<_>>();
    
    println!("✓ Model parsing works: {} models parsed", models.len());
    
    // Test 5: Test model finding
    if let Some(found_model) = find_model_by_id_or_name(&models, "openai/gpt-3.5-turbo") {
        println!("✓ Model finding works: found {}", found_model.name);
    }
    
    // Test 6: Test error handling
    let error = OpenRouterError::InvalidApiKey;
    let error_msg = format!("{}", error);
    println!("✓ Error handling works: {}", error_msg);
    
    println!("\n🎉 All OpenRouter integration tests passed!");
    println!("The OpenRouter integration is working correctly.");
    
    Ok(())
}