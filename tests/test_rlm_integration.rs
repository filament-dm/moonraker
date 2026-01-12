//! Integration tests for RLM with Rig provider
//!
//! These tests validate that the RLM works correctly with the Rig provider
//! using XML tag parsing (see README.md for why we use XML instead of structured output).

use moonraker::rlm::{RigProvider, Rlm};

#[cfg(feature = "integration")]
fn get_test_model() -> String {
    std::env::var("MOONRAKER_TEST_MODEL").unwrap_or_else(|_| "qwen3:30b".to_string())
}

const SYSTEM_PROMPT: &str = r#"You are a Lua programming assistant. Your task is to write Lua code to solve the user's request.

For each response, provide your output in the following XML format:

<comment>
Description of what the code does
</comment>

<code>
The Lua code to execute (no backticks needed)
</code>

<final>
Either "true" or "false" - use "true" when you have completed the task
</final>

The code should be syntactically correct Lua that can be executed immediately.
Use global variables (NO 'local' keyword) so state persists across multiple executions.
Use print() to output results.

Example:

<comment>
Creates a test string and splits it on newlines
</comment>

<code>
test_str = "line1\nline2\nline3"
for line in string.gmatch(test_str, "[^\n]+") do
    print(line)
end
</code>

<final>
true
</final>
"#;

/// Test 1: Fibonacci calculation
/// - Write a fibonacci function
/// - Calculate the 10th term
/// - Add 1000 to it
#[cfg(feature = "integration")]
#[tokio::test]
async fn test_rlm_fibonacci() {
    let model = get_test_model();
    // Create the provider with system prompt
    let provider = RigProvider::new_ollama_with_system(model.clone(), SYSTEM_PROMPT.to_string());

    // Create the RLM
    let prompt = "Write a Fibonacci function, calculate the 10th Fibonacci number, then add 1000 to it and print the result.".to_string();
    let llm_client = moonraker::environment::LlmClient::Ollama(model.clone());
    let mut rlm = Rlm::new(
        provider,
        prompt,
        String::new(), // No context needed
        model,
        llm_client,
    )
    .expect("Failed to create RLM");

    // Execute with max 5 iterations
    let mut iter = rlm.execute(5);
    let mut iteration = 0;
    let mut completed = false;

    while let Some(result) = iter.next().await {
        iteration += 1;

        match result {
            Ok(cell) => {
                println!("\n=== Iteration {iteration} ===");
                println!("Comment: {}", cell.comment);
                println!("Code: {}", cell.code);
                if let Some(output) = &cell.output {
                    println!("Output: {output}");
                }
                println!("Final: {}", cell.r#final);

                if cell.r#final {
                    completed = true;

                    // Verify the final output contains a number around 1055
                    // (10th Fibonacci number is 55, plus 1000 is 1055)
                    if let Some(output) = &cell.output {
                        assert!(
                            output.contains("1055"),
                            "Expected output to contain 1055, got: {output}"
                        );
                    } else {
                        panic!("Final cell should have output");
                    }

                    break;
                }
            }
            Err(e) => {
                panic!("Error in iteration {iteration}: {e}");
            }
        }
    }

    assert!(completed, "RLM should complete within 5 iterations");
}

/// Test 2: String splitting on line breaks
/// - Define a test string with 3 lines
/// - Split on newlines
/// - Print each line
#[cfg(feature = "integration")]
#[tokio::test]
async fn test_rlm_string_split() {
    let model = get_test_model();
    // Create the provider with system prompt
    let provider = RigProvider::new_ollama_with_system(model.clone(), SYSTEM_PROMPT.to_string());

    // Create the RLM
    let prompt = "Write a Lua program that defines a test string with 3 lines of text, splits the string on line breaks (newlines), and prints each line separately.".to_string();
    let llm_client = moonraker::environment::LlmClient::Ollama(model.clone());
    let mut rlm = Rlm::new(
        provider,
        prompt,
        String::new(), // No context needed
        model,
        llm_client,
    )
    .expect("Failed to create RLM");

    // Execute with max 3 iterations
    let mut iter = rlm.execute(3);
    let mut iteration = 0;
    let mut completed = false;

    while let Some(result) = iter.next().await {
        iteration += 1;

        match result {
            Ok(cell) => {
                println!("\n=== Iteration {iteration} ===");
                println!("Comment: {}", cell.comment);
                println!("Code: {}", cell.code);
                if let Some(output) = &cell.output {
                    println!("Output: {output}");
                }
                println!("Final: {}", cell.r#final);

                if cell.r#final {
                    completed = true;

                    // Verify the final output contains at least 3 lines
                    if let Some(output) = &cell.output {
                        let line_count = output.lines().count();
                        assert!(
                            line_count >= 3,
                            "Expected at least 3 lines of output, got: {line_count}"
                        );
                    } else {
                        panic!("Final cell should have output");
                    }

                    break;
                }
            }
            Err(e) => {
                panic!("Error in iteration {iteration}: {e}");
            }
        }
    }

    assert!(completed, "RLM should complete within 3 iterations");
}
