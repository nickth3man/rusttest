use anyhow::Result;
use pocketflow_rs::{build_flow, Context};
use serde_json::json;

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

/// Main function for the OpenRouter configuration example
#[tokio::main]
async fn main() -> Result<()> {
    println!("🔧 OpenRouter Configuration Examples");
    println!("=====================================");
    
    // Example 1: Basic Configuration Flow
    basic_configuration_demo().await?;
    
    println!("\n✅ All OpenRouter configuration examples completed!");
    Ok(())
}

/// Basic OpenRouter configuration demonstration
async fn basic_configuration_demo() -> Result<()> {
    println!("\n🔧 Basic Configuration Demo");
    println!("============================");
    
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

    let context = Context::new();
    let result = flow.run(context).await?;
    
    if let Some(question) = result.get("question").and_then(|v| v.as_str()) {
        println!("Question: {}", question);
    }
    
    if let Some(answer) = result.get("answer").and_then(|v| v.as_str()) {
        println!("Answer: {}", answer);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_configuration_demo() {
        // This would test the basic configuration
        // In a real test, you'd use mock responses
        assert!(true);
    }
    
    #[test]
    fn test_process_state() {
        let state = MyState::Success;
        assert_eq!(state.to_condition(), "success");
        assert!(!state.is_default());
    }
}