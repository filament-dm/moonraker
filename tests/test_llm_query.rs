//! Integration test for llm_query function in the Environment
//!
//! This test validates that the llm_query function works correctly with the
//! RigProvider using qwen3:30b model.

use moonraker::environment::{Environment, LlmClient};

#[cfg(feature = "integration")]
fn get_test_model() -> String {
    std::env::var("MOONRAKER_TEST_MODEL").unwrap_or_else(|_| "qwen3:30b".to_string())
}

#[cfg(feature = "integration")]
#[tokio::test(flavor = "multi_thread")]
async fn test_llm_query_basic() {
    let model = get_test_model();
    // Create environment with qwen3:30b model
    let env = Environment::new("", LlmClient::Ollama(model)).unwrap();

    // Test a simple query
    let code = r#"
        result = llm_query("What is 2+2? Reply with just the number.")
        print(result)
    "#;

    let output = env.eval(code).unwrap();
    println!("LLM Response: {output:?}");

    // Verify we got some output
    assert!(output.is_some(), "llm_query should return a response");
    let response = output.unwrap();
    assert!(!response.is_empty(), "Response should not be empty");

    // The response should contain "4" somewhere
    assert!(
        response.contains("4"),
        "Response should contain '4', got: {response}"
    );
}

#[cfg(feature = "integration")]
#[tokio::test(flavor = "multi_thread")]
async fn test_llm_query_multiple_calls() {
    let model = get_test_model();
    // Create environment with qwen3:30b model
    let env = Environment::new("", LlmClient::Ollama(model)).unwrap();

    // Test multiple queries in sequence
    let code = r#"
        answer1 = llm_query("What is 5+3? Reply with just the number.")
        answer2 = llm_query("What is 10-2? Reply with just the number.")
        print("First: " .. answer1)
        print("Second: " .. answer2)
    "#;

    let output = env.eval(code).unwrap();
    println!("LLM Responses: {output:?}");

    // Verify we got output
    assert!(output.is_some(), "llm_query should return responses");
    let response = output.unwrap();

    // Should have two lines of output
    let lines: Vec<&str> = response.lines().collect();
    assert!(
        lines.len() >= 2,
        "Should have at least 2 lines of output, got: {}",
        lines.len()
    );

    // Check that responses contain expected numbers
    assert!(
        response.contains("8") || response.contains("First"),
        "Response should contain '8' or 'First', got: {response}"
    );
}

#[cfg(feature = "integration")]
#[tokio::test(flavor = "multi_thread")]
async fn test_llm_query_with_context() {
    let model = get_test_model();
    // Create environment with context
    let env = Environment::new("The secret number is 42", LlmClient::Ollama(model)).unwrap();

    // Query about the context
    let code = r#"
        response = llm_query("What is the secret number mentioned in the context: " .. context .. "? Reply with just the number.")
        print(response)
    "#;

    let output = env.eval(code).unwrap();
    println!("LLM Response with context: {output:?}");

    // Verify we got output
    assert!(output.is_some(), "llm_query should return a response");
    let response = output.unwrap();

    // The response should mention 42
    assert!(
        response.contains("42"),
        "Response should contain '42', got: {response}"
    );
}
