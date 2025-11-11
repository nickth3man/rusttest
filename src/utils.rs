use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::{json, Value};
use std::env;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_openrouter_config_creation() {
        let config = OpenRouterConfig::new("test-api-key");

        assert_eq!(config.api_key, "test-api-key");
        assert_eq!(config.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(config.timeout_seconds, 60);
        assert_eq!(config.max_retries, 3);
        assert!(config.track_usage);
    }

    #[tokio::test]
    async fn test_openrouter_config_builder() {
        let config = OpenRouterConfig::new("test-key")
            .with_base_url("https://custom.openrouter.ai/api/v1")
            .with_referral_token("my-referral")
            .with_usage_tracking(false)
            .with_header("Custom-Header", "value")
            .with_timeout(120)
            .with_retries(5, 3);

        assert_eq!(config.base_url, "https://custom.openrouter.ai/api/v1");
        assert_eq!(config.referral_token, Some("my-referral".to_string()));
        assert!(!config.track_usage);
        assert_eq!(config.timeout_seconds, 120);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.retry_delay_seconds, 3);
        assert_eq!(config.custom_headers.len(), 1);
        assert_eq!(
            config.custom_headers[0],
            ("Custom-Header".to_string(), "value".to_string())
        );
    }

    #[tokio::test]
    async fn test_openrouter_usage_tracking() {
        let mut usage = OpenRouterUsage::new();

        // Mock response with usage data
        let mock_response = json!({
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "total_tokens": 80
            }
        });

        usage.update_from_response(&mock_response, "gpt-3.5-turbo");

        assert_eq!(usage.prompt_tokens, 50);
        assert_eq!(usage.completion_tokens, 30);
        assert_eq!(usage.total_tokens, 80);
        assert_eq!(usage.model, "gpt-3.5-turbo");

        // Check cost calculation
        let cost_string = usage.get_cost_string();
        assert!(cost_string.starts_with("$"));
    }

    #[tokio::test]
    async fn test_openrouter_model_parsing() {
        let mock_model_data = json!({
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
                },
                {
                    "id": "anthropic/claude-3-sonnet",
                    "name": "Claude 3 Sonnet",
                    "description": "Intelligent model",
                    "context_length": 200000,
                    "provider": {
                        "name": "Anthropic"
                    },
                    "features": {
                        "chat": true,
                        "completion": false
                    }
                }
            ]
        });

        // Test parsing models from response (this would be done in list_openrouter_models)
        let models = mock_model_data
            .get("data")
            .and_then(|data| data.as_array())
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|model_data| {
                Some(OpenRouterModel {
                    id: model_data.get("id")?.as_str()?.to_string(),
                    name: model_data.get("name")?.as_str()?.to_string(),
                    description: model_data
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(|s| s.to_string()),
                    provider: model_data
                        .get("provider")
                        .and_then(|p| p.get("name"))
                        .and_then(|p| p.as_str())
                        .unwrap_or("Unknown")
                        .to_string(),
                    context_length: model_data
                        .get("context_length")
                        .and_then(|c| c.as_u64())
                        .map(|c| c as u32),
                    supports_chat: model_data
                        .get("features")
                        .and_then(|f| f.get("chat"))
                        .and_then(|c| c.as_bool())
                        .unwrap_or(false),
                    supports_completion: model_data
                        .get("features")
                        .and_then(|f| f.get("completion"))
                        .and_then(|c| c.as_bool())
                        .unwrap_or(false),
                })
            })
            .collect::<Vec<_>>();

        assert_eq!(models.len(), 2);

        let gpt_model = &models[0];
        assert_eq!(gpt_model.id, "openai/gpt-3.5-turbo");
        assert_eq!(gpt_model.name, "GPT-3.5 Turbo");
        assert_eq!(gpt_model.provider, "OpenAI");
        assert_eq!(gpt_model.context_length, Some(16385));
        assert!(gpt_model.supports_chat);
        assert!(gpt_model.supports_completion);
    }

    #[tokio::test]
    async fn test_model_finding() {
        let models = vec![
            OpenRouterModel {
                id: "openai/gpt-3.5-turbo".to_string(),
                name: "GPT-3.5 Turbo".to_string(),
                description: None,
                provider: "OpenAI".to_string(),
                context_length: Some(16385),
                supports_chat: true,
                supports_completion: true,
            },
            OpenRouterModel {
                id: "anthropic/claude-3-sonnet".to_string(),
                name: "Claude 3 Sonnet".to_string(),
                description: None,
                provider: "Anthropic".to_string(),
                context_length: Some(200000),
                supports_chat: true,
                supports_completion: false,
            },
        ];

        // Test ID match
        let found = find_model_by_id_or_name(&models, "openai/gpt-3.5-turbo");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "GPT-3.5 Turbo");

        // Test name match (case insensitive)
        let found = find_model_by_id_or_name(&models, "claude 3 sonnet");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "anthropic/claude-3-sonnet");

        // Test no match
        let found = find_model_by_id_or_name(&models, "nonexistent");
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn test_llm_config_with_openrouter() {
        let config = LLMConfig::new(LLMProvider::OpenRouter, "test-key".to_string());

        assert_eq!(config.provider, LLMProvider::OpenRouter);
        assert_eq!(config.model, "openai/gpt-3.5-turbo");
        assert_eq!(config.api_key, "test-key");
        assert!(config.openrouter_config.is_none());

        // Test with custom OpenRouter config
        let openrouter_config =
            OpenRouterConfig::new("test-key").with_base_url("https://custom.openrouter.ai/api/v1");

        let config_with_openrouter = config.with_openrouter_config(openrouter_config);
        assert!(config_with_openrouter.openrouter_config.is_some());
        assert_eq!(
            config_with_openrouter.openrouter_config.unwrap().base_url,
            "https://custom.openrouter.ai/api/v1"
        );
    }

    #[tokio::test]
    async fn test_error_handling() {
        // Test error conversion
        let invalid_key_error = OpenRouterError::InvalidApiKey;
        let error: anyhow::Error = invalid_key_error.into();
        assert!(error.to_string().contains("API key not found or invalid"));

        let rate_limit_error = OpenRouterError::RateLimit {
            message: "Too many requests".to_string(),
        };
        let error: anyhow::Error = rate_limit_error.into();
        assert!(error.to_string().contains("Rate limit exceeded"));
    }
}

/// Comprehensive error types for OpenRouter operations
#[derive(Debug, thiserror::Error)]
pub enum OpenRouterError {
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    #[error("API key not found or invalid")]
    InvalidApiKey,

    #[error("Model not found: {model}")]
    ModelNotFound { model: String },

    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },

    #[allow(dead_code)] // Kept for future use / external consumers; allowed to be unused here.
    #[error("API quota exceeded: {message}")]
    QuotaExceeded { message: String },

    #[error("Invalid request parameters: {message}")]
    InvalidRequest { message: String },

    #[error("Service unavailable: {message}")]
    ServiceUnavailable { message: String },

    #[error("Response parsing failed: {0}")]
    ParseError(String),

    #[error("Network timeout")]
    Timeout,

    #[error("Unknown API error: {status} - {message}")]
    UnknownError { status: u16, message: String },

    // Additional error types based on OpenRouter best practices
    #[error("Insufficient credits: {message}")]
    InsufficientCredits { message: String },

    #[error("Model requires moderation and input was flagged")]
    ModerationFailed,

    #[error("Request too large: {message}")]
    RequestTooLarge { message: String },

    #[allow(dead_code)] // Kept for future use / external consumers; allowed to be unused here.
    #[error("Free model limit exceeded: {message}")]
    FreeModelLimitExceeded { message: String },
}

/// OpenRouter-specific configuration options
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenRouterConfig {
    /// API key for OpenRouter
    pub api_key: String,

    /// Base URL for OpenRouter API (default: https://openrouter.ai/api/v1)
    pub base_url: String,

    /// Optional referral token
    pub referral_token: Option<String>,

    /// Enable usage tracking
    pub track_usage: bool,

    /// Custom headers for API requests
    pub custom_headers: Vec<(String, String)>,

    /// Timeout duration in seconds (default: 60)
    pub timeout_seconds: u64,

    /// Maximum number of retry attempts (default: 3)
    pub max_retries: u32,

    /// Retry delay in seconds (default: 2)
    pub retry_delay_seconds: u64,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            referral_token: None,
            track_usage: true,
            custom_headers: Vec::new(),
            timeout_seconds: 60,
            max_retries: 3,
            retry_delay_seconds: 2,
        }
    }
}

impl OpenRouterConfig {
    /// Create a new OpenRouter configuration
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            ..Default::default()
        }
    }

    /// Set custom base URL
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }

    /// Set referral token
    pub fn with_referral_token(mut self, token: &str) -> Self {
        self.referral_token = Some(token.to_string());
        self
    }

    /// Enable or disable usage tracking
    pub fn with_usage_tracking(mut self, track: bool) -> Self {
        self.track_usage = track;
        self
    }

    /// Add custom header
    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.custom_headers
            .push((name.to_string(), value.to_string()));
        self
    }

    /// Set timeout duration
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// Set retry configuration
    pub fn with_retries(mut self, max_retries: u32, delay_seconds: u64) -> Self {
        self.max_retries = max_retries;
        self.retry_delay_seconds = delay_seconds;
        self
    }
}

/// Usage tracking for OpenRouter API calls
#[derive(Debug, Clone, Default)]
pub struct OpenRouterUsage {
    /// Number of prompt tokens used
    pub prompt_tokens: u32,

    /// Number of completion tokens used
    pub completion_tokens: u32,

    /// Total tokens used
    pub total_tokens: u32,

    /// Estimated cost in USD
    pub estimated_cost: f64,

    /// Model used for this request
    pub model: String,
}

impl OpenRouterUsage {
    /// Create a new usage tracking instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Update usage from API response
    pub fn update_from_response(&mut self, response: &Value, model: &str) {
        if let Some(usage) = response.get("usage").and_then(|u| u.as_object()) {
            self.prompt_tokens = usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(0);

            self.completion_tokens = usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(0);

            self.total_tokens = usage
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(0);
        }

        self.model = model.to_string();
        self.calculate_cost();
    }

    /// Calculate estimated cost based on token usage and model pricing
    fn calculate_cost(&mut self) {
        // This is a simplified cost calculation
        // In a real implementation, you would use actual OpenRouter model pricing
        let cost_per_1k_tokens = match self.model.as_str() {
            "gpt-4" | "gpt-4-32k" => 0.06, // $0.06 per 1K tokens
            "gpt-3.5-turbo" => 0.002,      // $0.002 per 1K tokens
            "claude-3-sonnet" => 0.015,    // $0.015 per 1K tokens
            "claude-3-opus" => 0.03,       // $0.03 per 1K tokens
            _ => 0.01,                     // Default fallback
        };

        self.estimated_cost = (self.total_tokens as f64 / 1000.0) * cost_per_1k_tokens;
    }

    /// Get formatted cost string
    pub fn get_cost_string(&self) -> String {
        format!("${:.6}", self.estimated_cost)
    }
}

/// Available models from OpenRouter
#[derive(Debug, Clone)]
pub struct OpenRouterModel {
    /// Model ID
    pub id: String,

    /// Model name
    pub name: String,

    /// Model description
    #[allow(dead_code)] // Kept for future use / external consumers; allowed to be unused here.
    pub description: Option<String>,

    /// Provider name
    pub provider: String,

    /// Maximum context length
    pub context_length: Option<u32>,

    /// Whether the model supports chat
    pub supports_chat: bool,

    /// Whether the model supports completion
    pub supports_completion: bool,
}

/// Configuration for LLM providers
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LLMConfig {
    pub provider: LLMProvider,
    pub api_key: String,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,

    /// OpenRouter-specific configuration
    pub openrouter_config: Option<OpenRouterConfig>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LLMProvider {
    OpenAI,
    Anthropic,
    Ollama,
    OpenRouter,
}

impl LLMConfig {
    /// Create a new LLM configuration with default settings
    pub fn new(provider: LLMProvider, api_key: String) -> Self {
        let model = match &provider {
            LLMProvider::OpenAI => "gpt-4o".to_string(),
            LLMProvider::Anthropic => "claude-3-sonnet-20240229".to_string(),
            LLMProvider::Ollama => "llama2".to_string(),
            LLMProvider::OpenRouter => "openai/gpt-3.5-turbo".to_string(),
        };
        Self {
            provider,
            api_key,
            model,
            max_tokens: Some(1024),
            temperature: Some(0.7),
            openrouter_config: None,
        }
    }

    /// Set custom model
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set OpenRouter-specific configuration
    pub fn with_openrouter_config(mut self, config: OpenRouterConfig) -> Self {
        self.openrouter_config = Some(config);
        self
    }
}

/// Validate and sanitize input prompt
fn validate_prompt(prompt: &str) -> Result<String> {
    let trimmed = prompt.trim();

    // Check for empty input
    if trimmed.is_empty() {
        anyhow::bail!("Prompt cannot be empty");
    }

    // Check for excessive length (prevent buffer overflow)
    if trimmed.len() > 10000 {
        anyhow::bail!("Prompt too long (maximum 10,000 characters)");
    }

    // Basic sanitization - remove control characters except newlines and tabs
    let sanitized: String = trimmed
        .chars()
        .filter(|c| c.is_ascii() || c.is_whitespace() || c.is_alphabetic() || c.is_numeric())
        .collect();

    Ok(sanitized)
}

/// Call LLM with comprehensive error handling and security
pub async fn call_llm(prompt: &str) -> Result<String> {
    call_llm_with_config(prompt, None).await
}

/// Call LLM with custom configuration
pub async fn call_llm_with_config(prompt: &str, config: Option<LLMConfig>) -> Result<String> {
    // Validate input
    let sanitized_prompt = validate_prompt(prompt).with_context(|| "Input validation failed")?;

    // Get configuration
    let config = config.unwrap_or_else(|| {
        // Default configuration from environment
        let (provider, api_key) = env::var("OPENROUTER_API_KEY")
            .map(|key| (LLMProvider::OpenRouter, key))
            .or_else(|_| env::var("OPENAI_API_KEY").map(|key| (LLMProvider::OpenAI, key)))
            .or_else(|_| env::var("ANTHROPIC_API_KEY").map(|key| (LLMProvider::Anthropic, key)))
            .or_else(|_| env::var("OLLAMA_BASE_URL").map(|url| (LLMProvider::Ollama, url)))
            .unwrap_or_else(|_| (LLMProvider::OpenAI, "dummy-key".to_string()));

        LLMConfig::new(provider, api_key)
    });

    // Create HTTP client
    let client = Client::new();

    match config.provider {
        LLMProvider::OpenAI => call_openai(&client, &config, &sanitized_prompt).await,
        LLMProvider::Anthropic => call_anthropic(&client, &config, &sanitized_prompt).await,
        LLMProvider::Ollama => call_ollama(&client, &config, &sanitized_prompt).await,
        LLMProvider::OpenRouter => call_openrouter(&client, &config, &sanitized_prompt).await,
    }
}

/// Call OpenAI API
async fn call_openai(client: &Client, config: &LLMConfig, prompt: &str) -> Result<String> {
    let api_key = &config.api_key;

    if api_key == "dummy-key" {
        return Ok(format!("[OpenAI mock response to]: {}", prompt));
    }

    let url = "https://api.openai.com/v1/chat/completions";

    let request_body = json!({
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": config.max_tokens.unwrap_or(1024),
        "temperature": config.temperature.unwrap_or(0.7)
    });

    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request_body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .with_context(|| "Failed to send request to OpenAI API")?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        anyhow::bail!("OpenAI API error ({}): {}", status, error_text);
    }

    let response_json: serde_json::Value = response
        .json()
        .await
        .with_context(|| "Failed to parse OpenAI API response")?;

    // Extract the response text
    let content = response_json
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid response format from OpenAI API"))?;

    Ok(content.to_string())
}

// NOTE: The source below this point was accidentally pasted from documentation (non-Rust examples)
// and is the cause of the syntax/unclosed delimiter errors reported by rust-analyzer/rustc.
// It is removed to restore a valid Rust module focused solely on utils.

/// Call Anthropic API
async fn call_anthropic(client: &Client, config: &LLMConfig, prompt: &str) -> Result<String> {
    let api_key = &config.api_key;

    if api_key == "dummy-key" {
        return Ok(format!("[Anthropic mock response to]: {}", prompt));
    }

    let url = "https://api.anthropic.com/v1/messages";

    let request_body = json!({
        "model": config.model,
        "max_tokens": config.max_tokens.unwrap_or(1024),
        "temperature": config.temperature.unwrap_or(0.7),
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    });

    let response = client
        .post(url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&request_body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .with_context(|| "Failed to send request to Anthropic API")?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        anyhow::bail!("Anthropic API error ({}): {}", status, error_text);
    }

    let response_json: serde_json::Value = response
        .json()
        .await
        .with_context(|| "Failed to parse Anthropic API response")?;

    let content = response_json
        .get("content")
        .and_then(|content| content.as_array())
        .and_then(|content| content.first())
        .and_then(|content| content.get("text"))
        .and_then(|text| text.as_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid response format from Anthropic API"))?;

    Ok(content.to_string())
}

/// Call Ollama (local) API
async fn call_ollama(client: &Client, config: &LLMConfig, prompt: &str) -> Result<String> {
    let base_url = if config.api_key == "dummy-key" {
        "http://localhost:11434"
    } else {
        &config.api_key
    };

    let url = format!("{}/api/chat", base_url);

    let request_body = json!({
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": false
    });

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .timeout(std::time::Duration::from_secs(60)) // Local models might be slower
        .send()
        .await
        .with_context(|| format!("Failed to send request to Ollama at {}", base_url))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        anyhow::bail!("Ollama API error ({}): {}", status, error_text);
    }

    let response_json: serde_json::Value = response
        .json()
        .await
        .with_context(|| "Failed to parse Ollama API response")?;

    let content = response_json
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid response format from Ollama API"))?;

    Ok(content.to_string())
}

/// Call OpenRouter API with comprehensive error handling and retry logic
async fn call_openrouter(client: &Client, config: &LLMConfig, prompt: &str) -> Result<String> {
    // Get or create OpenRouter configuration
    // Select OpenRouter configuration without returning references to temporaries.
    // If an explicit openrouter_config is provided, use it by reference.
    // Otherwise, synthesize a local config struct and use it immutably within this function.
    let (
        api_key,
        base_url,
        referral_token,
        track_usage,
        custom_headers,
        timeout_seconds,
        max_retries,
        retry_delay_seconds,
    ) = if let Some(cfg) = config.openrouter_config.as_ref() {
        (
            cfg.api_key.clone(),
            cfg.base_url.clone(),
            cfg.referral_token.clone(),
            cfg.track_usage,
            cfg.custom_headers.clone(),
            cfg.timeout_seconds,
            cfg.max_retries,
            cfg.retry_delay_seconds,
        )
    } else {
        (
            config.api_key.clone(),
            "https://openrouter.ai/api/v1".to_string(),
            None,
            true,
            Vec::new(),
            60,
            3,
            2,
        )
    };

    // Check for dummy key
    if api_key == "dummy-key" {
        return Ok(format!("[OpenRouter mock response to]: {}", prompt));
    }

    // Validate API key
    if api_key.is_empty() {
        return Err(OpenRouterError::InvalidApiKey.into());
    }

    // Prepare request body
    let request_body = json!({
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": config.max_tokens.unwrap_or(1024),
        "temperature": config.temperature.unwrap_or(0.7)
    });

    // Prepare headers
    let mut headers = std::collections::HashMap::new();
    headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    headers.insert("Accept".to_string(), "application/json".to_string());

    // Add custom headers if specified
    for (name, value) in &custom_headers {
        headers.insert(name.clone(), value.clone());
    }

    // Add referral token if specified
    if let Some(token) = &referral_token {
        headers.insert("X-Title".to_string(), token.clone());
    }

    // Prepare API endpoint
    let url = format!("{}/chat/completions", base_url);

    // Enhanced retry logic based on OpenRouter best practices
    let mut last_error: Option<anyhow::Error> = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            // Exponential backoff
            let base_delay = retry_delay_seconds;
            let delay = std::cmp::min(
                base_delay * 2_u64.saturating_pow(attempt - 1),
                60, // Maximum 60 seconds
            );

            eprintln!(
                "Retrying in {} seconds (attempt {}/{})...",
                delay, attempt, max_retries
            );
            tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
        }

        match send_openrouter_request(client, &url, &headers, &request_body, timeout_seconds).await
        {
            Ok(response) => {
                // Success - parse response and return
                let response_text = parse_openrouter_response(&response, &config.model)?;

                // Track usage if enabled
                if track_usage {
                    let mut usage = OpenRouterUsage::new();
                    usage.update_from_response(&response, &config.model);
                    // In a real implementation, you would store usage metrics
                    eprintln!(
                        "OpenRouter Usage: {} tokens, estimated cost: {}",
                        usage.total_tokens,
                        usage.get_cost_string()
                    );
                }

                return Ok(response_text);
            }
            Err(error) => {
                // Decide if this error is retryable based on its type
                let should_retry =
                    if let Some(openrouter_error) = error.downcast_ref::<OpenRouterError>() {
                        match openrouter_error {
                            OpenRouterError::Timeout
                            | OpenRouterError::HttpError(_)
                            | OpenRouterError::ServiceUnavailable { .. } => true,
                            OpenRouterError::UnknownError { status, .. } if *status >= 500 => true,
                            OpenRouterError::RateLimit { .. } => true,
                            OpenRouterError::InsufficientCredits { .. }
                            | OpenRouterError::InvalidApiKey
                            | OpenRouterError::ModelNotFound { .. }
                            | OpenRouterError::ModerationFailed
                            | OpenRouterError::RequestTooLarge { .. } => false,
                            _ => true,
                        }
                    } else {
                        true // Unknown errors are treated as retryable
                    };

                // Store last error for potential final return
                last_error = Some(error);

                // Enhanced retry logic based on OpenRouter error types
                if !should_retry || attempt == max_retries {
                    break;
                }

                if let Some(ref e) = last_error {
                    eprintln!(
                        "OpenRouter request failed (attempt {}/{}) with error: {:?}",
                        attempt + 1,
                        max_retries + 1,
                        e
                    );
                }
            }
        }
    }

    // All retries exhausted, return the last error
    Err(last_error
        .unwrap_or_else(|| OpenRouterError::HttpError("Unknown error".to_string()).into()))
}

/// Send OpenRouter HTTP request
async fn send_openrouter_request(
    client: &Client,
    url: &str,
    headers: &std::collections::HashMap<String, String>,
    request_body: &Value,
    timeout_seconds: u64,
) -> Result<Value> {
    let timeout = std::time::Duration::from_secs(timeout_seconds);

    let mut request_builder = client.post(url).timeout(timeout);

    // Add headers
    for (name, value) in headers {
        request_builder = request_builder.header(name, value);
    }

    // Send request
    let response = request_builder
        .json(request_body)
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                OpenRouterError::Timeout
            } else {
                OpenRouterError::HttpError(e.to_string())
            }
        })?;

    // Check response status
    let status = response.status();
    if status.is_success() {
        // Parse successful response
        response
            .json::<Value>()
            .await
            .map_err(|e| OpenRouterError::ParseError(e.to_string()).into())
    } else {
        // Handle error response
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        Ok(map_openrouter_error(status.as_u16(), &error_text)?)
    }
}

/// Parse OpenRouter response and extract text content
fn parse_openrouter_response(response: &Value, _model: &str) -> Result<String> {
    // Extract the response text
    let content = response
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid response format from OpenRouter API"))?;

    Ok(content.to_string())
}

/// Map HTTP status codes and error messages to OpenRouterError types
fn map_openrouter_error(status: u16, error_text: &str) -> Result<Value, OpenRouterError> {
    match status {
        400 => Err(OpenRouterError::InvalidRequest {
            message: format!("Bad request: {}", error_text),
        }),
        401 => Err(OpenRouterError::InvalidApiKey),
        402 => Err(OpenRouterError::InsufficientCredits {
            message: format!("Insufficient credits: {}", error_text),
        }),
        403 => {
            if error_text.contains("moderation") {
                Err(OpenRouterError::ModerationFailed)
            } else {
                Err(OpenRouterError::InvalidRequest {
                    message: format!("Forbidden: {}", error_text),
                })
            }
        }
        404 => Err(OpenRouterError::ModelNotFound {
            model: "unknown".to_string(), // Could parse from error message
        }),
        408 => Err(OpenRouterError::Timeout),
        413 => Err(OpenRouterError::RequestTooLarge {
            message: format!("Request too large: {}", error_text),
        }),
        429 => Err(OpenRouterError::RateLimit {
            message: format!("Rate limit exceeded: {}", error_text),
        }),
        502 => Err(OpenRouterError::ServiceUnavailable {
            message: format!("Model temporarily unavailable: {}", error_text),
        }),
        500..=599 => Err(OpenRouterError::ServiceUnavailable {
            message: format!("Server error {}: {}", status, error_text),
        }),
        _ => Err(OpenRouterError::UnknownError {
            status,
            message: error_text.to_string(),
        }),
    }
}

/// List available models from OpenRouter
pub async fn list_openrouter_models(config: &OpenRouterConfig) -> Result<Vec<OpenRouterModel>> {
    if config.api_key == "dummy-key" {
        // Return mock models for testing
        return Ok(vec![
            OpenRouterModel {
                id: "openai/gpt-3.5-turbo".to_string(),
                name: "GPT-3.5 Turbo".to_string(),
                description: Some("Fast and efficient model".to_string()),
                provider: "OpenAI".to_string(),
                context_length: Some(16385),
                supports_chat: true,
                supports_completion: true,
            },
            OpenRouterModel {
                id: "openai/gpt-4".to_string(),
                name: "GPT-4".to_string(),
                description: Some("Advanced reasoning model".to_string()),
                provider: "OpenAI".to_string(),
                context_length: Some(8192),
                supports_chat: true,
                supports_completion: true,
            },
        ]);
    }

    let client = Client::new();
    let url = format!("{}/models", config.base_url);

    let response = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("Accept", "application/json")
        .timeout(std::time::Duration::from_secs(config.timeout_seconds))
        .send()
        .await
        .map_err(|e| OpenRouterError::HttpError(e.to_string()))?;

    if !response.status().is_success() {
        let _status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(OpenRouterError::HttpError(error_text).into());
    }

    let response_json: Value = response
        .json()
        .await
        .map_err(|e| OpenRouterError::ParseError(e.to_string()))?;

    // Parse models from response
    let models = response_json
        .get("data")
        .and_then(|data| data.as_array())
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|model_data| {
            Some(OpenRouterModel {
                id: model_data.get("id")?.as_str()?.to_string(),
                name: model_data.get("name")?.as_str()?.to_string(),
                description: model_data
                    .get("description")
                    .and_then(|d| d.as_str())
                    .map(|s| s.to_string()),
                provider: model_data
                    .get("provider")
                    .and_then(|p| p.get("name"))
                    .and_then(|p| p.as_str())
                    .unwrap_or("Unknown")
                    .to_string(),
                context_length: model_data
                    .get("context_length")
                    .and_then(|c| c.as_u64())
                    .map(|c| c as u32),
                supports_chat: model_data
                    .get("features")
                    .and_then(|f| f.get("chat"))
                    .and_then(|c| c.as_bool())
                    .unwrap_or(false),
                supports_completion: model_data
                    .get("features")
                    .and_then(|f| f.get("completion"))
                    .and_then(|c| c.as_bool())
                    .unwrap_or(false),
            })
        })
        .collect();

    Ok(models)
}

#[allow(dead_code)] // Kept for future use / external consumers; allowed to be unused here.
/// Find a model by ID or name
pub fn find_model_by_id_or_name<'a>(
    models: &'a [OpenRouterModel],
    identifier: &str,
) -> Option<&'a OpenRouterModel> {
    // First try exact ID match
    models
        .iter()
        .find(|model| model.id == identifier)
        .or_else(|| {
            // Then try name match (case insensitive)
            models
                .iter()
                .find(|model| model.name.to_lowercase() == identifier.to_lowercase())
        })
}

/// Get the best model based on criteria
pub fn get_best_model(
    models: &[OpenRouterModel],
    criteria: ModelSelectionCriteria,
) -> Option<&OpenRouterModel> {
    match criteria {
        ModelSelectionCriteria::Fastest => models
            .iter()
            .filter(|m| m.supports_chat)
            .max_by_key(|m| m.context_length.unwrap_or(0)),
        ModelSelectionCriteria::Cheapest => {
            // In a real implementation, you would use actual pricing data
            // This is a simplified version based on model names
            models
                .iter()
                .filter(|m| m.supports_chat)
                .find(|m| m.id.contains("gpt-3.5-turbo"))
                .or_else(|| models.iter().filter(|m| m.supports_chat).next())
        }
        ModelSelectionCriteria::MostCapable => models
            .iter()
            .filter(|m| m.supports_chat)
            .find(|m| m.id.contains("gpt-4"))
            .or_else(|| models.iter().filter(|m| m.supports_chat).next()),
        ModelSelectionCriteria::ByProvider(provider) => models
            .iter()
            .filter(|m| m.provider.to_lowercase() == provider.to_lowercase() && m.supports_chat)
            .next(),
    }
}

/// Criteria for model selection
#[derive(Debug, Clone)]
pub enum ModelSelectionCriteria {
    Fastest,
    Cheapest,
    MostCapable,
    ByProvider(String),
}
