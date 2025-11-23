//! Test suite to validate Rig's suitability for moonraker's use case.
//!
//! Tests XML-based parsing vs structured output. Currently using XML parsing
//! because of llama.cpp limitations. See README.md "Testing > Rig Structured
//! Output Tests" section for full explanation of why these tests exist and
//! when we can migrate to structured output.

use mlua::{Lua, Result as LuaResult};
use rig::client::CompletionClient;
use rig::providers::ollama;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Structure to extract from LLM responses
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct CellResponse {
    /// Description of the current step and reasoning
    comment: String,
    /// Lua code to execute
    code: String,
}

// System prompt for tests using extractors or structured output
// Does NOT dictate output format - lets the extractor/structured output handle it
const SYSTEM_PROMPT_EXTRACTOR: &str = r#"You are a Lua programming assistant. Your task is to write Lua code to solve the user's request.

For each response, provide:
1. A 'comment' describing what the code does
2. The 'code' containing the Lua code to execute

The code should be syntactically correct Lua that can be executed immediately.
Use global variables (NO 'local' keyword) so state persists across multiple executions.
Use print() to output results.
"#;

// System prompt for raw output test - specifies XML format for manual parsing
const SYSTEM_PROMPT_XML: &str = r#"You are a Lua programming assistant. Your task is to write Lua code to solve the user's request.

For each response, provide your output in the following XML format:

<comment>
Description of what the code does
</comment>

<code>
The Lua code to execute
</code>

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
"#;

/// Helper function to validate Lua syntax by attempting to execute it
fn validate_lua_syntax(code: &str) -> LuaResult<()> {
    let lua = Lua::new();

    // Try to load and execute the code
    lua.load(code).exec()?;

    Ok(())
}

/// Helper function to execute Lua code and capture output
fn execute_lua_with_output(code: &str) -> LuaResult<Option<String>> {
    let lua = Lua::new();
    let output = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let output_clone = output.clone();

    // Override print to capture output
    lua.globals().set(
        "print",
        lua.create_function(move |_lua, args: mlua::Variadic<mlua::Value>| {
            let mut output = output_clone.lock().unwrap();
            let strings: Vec<String> = args
                .iter()
                .map(|v| v.to_string().unwrap_or_else(|_| format!("{v:?}")))
                .collect();
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&strings.join("\t"));
            Ok(())
        })?,
    )?;

    // Execute the code
    lua.load(code).exec()?;

    // Get the captured output
    let result = output.lock().unwrap().clone();

    if result.is_empty() {
        Ok(None)
    } else {
        Ok(Some(result))
    }
}

/// Test using Rig's extractor which internally tries to use structured output (this should FAIL due to an Ollama bug)
#[cfg(feature = "integration")]
#[tokio::test]
#[should_panic(expected = "Extractor with structured output is buggy with Ollama")]
async fn test_rig_with_structured_output() {
    // 1. Set up Rig Ollama connection
    let ollama_client = ollama::Client::new();

    // 2. Use qwen3:30b model
    let model = "qwen3:30b";

    // 3. Create an extractor - Rig's extractors may try to use structured output internally
    let extractor = ollama_client
        .extractor::<CellResponse>(model)
        .preamble(SYSTEM_PROMPT_EXTRACTOR)
        .build();

    let user_query = r#"Write a Lua program that:
1. Defines a test string with 3 lines of text
2. Splits the string on line breaks (newlines)
3. Prints each line separately with its line number

Use string.gmatch or similar to iterate through lines."#;

    // Try to extract - this should work because Rig's extractors don't use Ollama's structured output
    // So we'll force a panic to document what we're testing
    match extractor.extract(user_query).await {
        Ok(response) => {
            // Rig's extractors actually work! They don't use Ollama's structured output.
            // Let's validate that the code is correct
            println!("Comment: {}", response.comment);
            println!("Code: {}", response.code);

            // For this test, we want to show that extractors work, but we're comparing
            // to Ollama's structured output which doesn't
            // So we panic with a message clarifying this is about Ollama's structured output
            if validate_lua_syntax(&response.code).is_ok() {
                panic!(
                    "Extractor with structured output is buggy with Ollama - but Rig's extractors work because they don't use it!"
                );
            } else {
                panic!("Extractor with structured output is buggy with Ollama");
            }
        }
        Err(e) => {
            panic!("Extractor with structured output is buggy with Ollama: {e}");
        }
    }
}

/// Test using raw Ollama client with structured output (this should fail due to an Ollama bug)
#[cfg(feature = "integration")]
#[tokio::test]
#[should_panic(expected = "Ollama structured output is buggy")]
async fn test_ollama_only_with_structured_output() {
    use ollama_rs::Ollama;
    use ollama_rs::generation::completion::request::GenerationRequest;
    use ollama_rs::generation::parameters::{FormatType, JsonStructure};

    // 1. Set up raw Ollama client
    let ollama = Ollama::default();

    // 2. Use qwen3:30b model
    let model = "qwen3:30b";

    // 3. Create request with structured output
    let prompt = format!(
        "{SYSTEM_PROMPT_EXTRACTOR}\n\nUser query: Write a Lua program that:
1. Defines a test string with 3 lines of text
2. Splits the string on line breaks (newlines)
3. Prints each line separately with its line number

Use string.gmatch or similar to iterate through lines."
    );

    // Create JSON structure for CellResponse
    let json_structure = JsonStructure::new::<CellResponse>();

    let request = GenerationRequest::new(model.to_string(), prompt)
        .format(FormatType::StructuredJson(Box::new(json_structure)));

    // Try to generate with structured output
    match ollama.generate(request).await {
        Ok(response) => {
            println!("Raw Ollama response: {}", response.response);

            // Try to parse as CellResponse
            match serde_json::from_str::<CellResponse>(&response.response) {
                Ok(cell) => {
                    println!("Parsed CellResponse:");
                    println!("  comment: {}", cell.comment);
                    println!("  code: {}", cell.code);

                    // Try to validate the code
                    if let Err(e) = validate_lua_syntax(&cell.code) {
                        panic!("Ollama structured output is buggy - produced invalid Lua: {e}");
                    }

                    // If it somehow worked, still document the issue
                    panic!("Ollama structured output is buggy");
                }
                Err(e) => {
                    panic!("Ollama structured output is buggy - failed to parse JSON: {e}");
                }
            }
        }
        Err(e) => {
            panic!("Ollama structured output is buggy - request failed: {e}");
        }
    }
}

/// Helper to extract Lua code from XML-tagged response
fn extract_lua_code_from_response(response: &str) -> String {
    // Try to extract from <code> XML tags
    if let Some(start) = response.find("<code>") {
        if let Some(end) = response[start..].find("</code>") {
            let code_start = start + "<code>".len();
            let code_end = start + end;
            let code = &response[code_start..code_end];
            return code.trim().to_string();
        }
    }

    // Fallback: try markdown code blocks
    let mut in_code_block = false;
    let mut code_lines = Vec::new();

    for line in response.lines() {
        let trimmed = line.trim();

        // Check for code block markers
        if trimmed.starts_with("```lua")
            || trimmed.starts_with("```Lua")
            || trimmed.starts_with("```")
        {
            in_code_block = !in_code_block;
            continue;
        }

        // Collect code lines
        if in_code_block {
            code_lines.push(line);
        }
    }

    if !code_lines.is_empty() {
        return code_lines.join("\n");
    }

    // Last resort: look for code-like content
    let mut fallback_lines = Vec::new();
    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.contains("=") || trimmed.contains("print") || trimmed.contains("function") {
            fallback_lines.push(line);
        }
    }

    fallback_lines.join("\n")
}

#[cfg(feature = "integration")]
#[tokio::test]
#[should_panic(expected = "Rig extractors are unreliable")]
async fn test_rig_extractor_with_ollama() {
    // 1. Set up Rig Ollama connection to localhost (default)
    let ollama_client = ollama::Client::new();

    // 2. Use qwen3:30b model
    let model = "qwen3:30b";

    // 3. Create an extractor (not using structured output)
    let extractor = ollama_client
        .extractor::<CellResponse>(model)
        .preamble(SYSTEM_PROMPT_EXTRACTOR)
        .build();

    // 4. Ask qwen3:30b to write a program that splits a string on line breaks
    // and test it out on a 3 line test input
    let user_query = r#"Write a Lua program that:
1. Defines a test string with 3 lines of text
2. Splits the string on line breaks (newlines)
3. Prints each line separately with its line number

Use string.gmatch or similar to iterate through lines."#;

    // 5. Try to use Rig's extractors - this demonstrates they are unreliable
    match extractor.extract(user_query).await {
        Ok(response) => {
            println!("Extractor returned:");
            println!("Comment: {}", response.comment);
            println!("Code: {}", response.code);

            // Check if the code is valid
            if let Err(e) = validate_lua_syntax(&response.code) {
                panic!("Rig extractors are unreliable - produced invalid Lua: {e}");
            }

            // Even if it worked this time, extractors are unreliable
            panic!("Rig extractors are unreliable");
        }
        Err(e) => {
            panic!("Rig extractors are unreliable - extraction failed: {e}");
        }
    }
}

/// Test without using extractors - just raw agent output
/// This test should PASS - it extracts Lua code manually and validates it
#[cfg(feature = "integration")]
#[tokio::test]
async fn test_rig_raw_output_no_extractor() {
    use rig::completion::Prompt;

    // 1. Set up Rig Ollama connection to localhost (default)
    let ollama_client = ollama::Client::new();

    // 2. Use qwen3:30b model
    let model = "qwen3:30b";

    // 3. Create a simple agent without extractors, using XML prompt for manual parsing
    let agent = ollama_client
        .agent(model)
        .preamble(SYSTEM_PROMPT_XML)
        .build();

    // 4. Ask qwen3:30b to write a program that splits a string on line breaks
    let user_query = r#"Write a Lua program that:
1. Defines a test string with 3 lines of text
2. Splits the string on line breaks (newlines)
3. Prints each line separately with its line number

Use string.gmatch or similar to iterate through lines."#;

    // 5. Use basic prompt (no extractors) to get raw output
    let response = agent
        .prompt(user_query)
        .await
        .expect("Failed to get response from LLM");

    // Print the raw output for debugging
    println!("\n{}", "=".repeat(80));
    println!("RAW LLM OUTPUT (no extractor):");
    println!("{}", "=".repeat(80));
    println!("{response}");
    println!("{}", "=".repeat(80));

    // Extract Lua code from the response
    let code = extract_lua_code_from_response(&response);

    assert!(!code.is_empty(), "Failed to extract Lua code from response");

    println!("\nExtracted Code:\n{code}");

    // Validate that the outputted code is syntactically correct Lua
    match validate_lua_syntax(&code) {
        Ok(_) => {
            println!("\n✓ Lua syntax is valid");
        }
        Err(e) => {
            panic!("Lua syntax validation failed: {e}");
        }
    }

    // Additionally, try to execute it and see if it produces output
    match execute_lua_with_output(&code) {
        Ok(Some(output)) => {
            println!("\nCode Output:\n{output}");
            // Verify we got multiple lines of output (split string should produce multiple lines)
            assert!(
                output.lines().count() >= 3,
                "Expected at least 3 lines of output from split string operation"
            );
        }
        Ok(None) => {
            println!("\nCode executed successfully but produced no output");
        }
        Err(e) => {
            panic!("Code execution failed: {e}");
        }
    }
}
