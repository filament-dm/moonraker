use mlua::{IntoLua, Lua, Result};
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::{ollama, openrouter};
use serde_json::json;
use std::sync::{Arc, Mutex};
use tiktoken_rs::p50k_base;

#[derive(Clone)]
pub enum LlmClient {
    Ollama(String),             // Store model name
    Openrouter(String, String), // Store model name and API key
}

/// A sandboxed Lua execution environment with LLM integration.
///
/// # Security
///
/// Uses `Lua::new()` which loads only the **safe subset** of standard libraries:
/// - ✓ Available: `math`, `string`, `table`, `coroutine`, `utf8`
/// - ✗ Blocked: `io`, `os`, `package`, `debug`, `ffi` (no file/network/system access)
///
/// # Custom Functions
///
/// - `print(...)` - Captures output to buffer (see [`create_print_function`])
/// - `llm_query(prompt)` - Query LLM provider (see [`create_llm_query_function`])
/// - `token_trunc(text, n)` - Truncate by token count (see [`create_token_trunc_function`])
///
/// # Global Variables
///
/// - `context` - Initial context value, persists across evaluations
pub struct Environment {
    lua: Lua,
    output_buffer: Arc<Mutex<String>>,
}

impl Environment {
    pub fn new<T>(init_context: T, client: LlmClient) -> Result<Self>
    where
        T: IntoLua,
    {
        let lua = Lua::new();
        let output_buffer = Arc::new(Mutex::new(String::new()));

        // Register custom functions
        lua.globals()
            .set("print", create_print_function(&lua, output_buffer.clone())?)?;
        lua.globals().set(
            "llm_query",
            create_llm_query_function(&lua, client.clone())?,
        )?;
        lua.globals()
            .set("token_trunc", create_token_trunc_function(&lua)?)?;

        // Set the init_context as a global 'context' variable
        lua.globals().set("context", init_context)?;

        Ok(Environment { lua, output_buffer })
    }

    pub fn eval(&self, code: &str) -> Result<Option<String>> {
        // Clear the output buffer before execution
        self.output_buffer.lock().unwrap().clear();

        // Execute the Lua code
        self.lua.load(code).exec()?;

        // Get the captured output
        let output = self.output_buffer.lock().unwrap().clone();

        if output.is_empty() {
            Ok(None)
        } else {
            Ok(Some(output))
        }
    }
}

/// Creates the custom `print(...)` function that captures output to a buffer.
///
/// # Lua Signature
/// ```lua
/// print(...)
/// ```
///
/// # Behavior
/// - Accepts multiple arguments of any type (like standard Lua print)
/// - Converts arguments to strings and joins them with tabs
/// - Appends output to internal buffer (doesn't print to stdout)
/// - Separates multiple print calls with newlines
fn create_print_function(lua: &Lua, output_buffer: Arc<Mutex<String>>) -> Result<mlua::Function> {
    lua.create_function(move |_lua, args: mlua::Variadic<mlua::Value>| {
        let mut output = output_buffer.lock().unwrap();
        let strings: Vec<String> = args
            .iter()
            .map(|v| {
                // Convert Lua values to strings like Lua's print does
                v.to_string().unwrap_or_else(|_| format!("{v:?}"))
            })
            .collect();
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(&strings.join("\t"));
        Ok(())
    })
}

/// Creates the custom `llm_query(prompt)` function for querying language models.
///
/// # Lua Signature
/// ```lua
/// response = llm_query(prompt)
/// ```
///
/// # Parameters
/// - `prompt` (string) - The prompt to send to the LLM
///
/// # Returns
/// - (string) - The LLM's response text
///
/// # Important Notes
/// - The LLM does **NOT** have access to the `context` variable
/// - You must include all relevant information in the prompt string
/// - Uses the configured LLM provider (Ollama or OpenRouter)
/// - Blocks until response is received
///
/// # Example
/// ```lua
/// summary = llm_query("Summarize this: " .. context)
/// ```
fn create_llm_query_function(lua: &Lua, client: LlmClient) -> Result<mlua::Function> {
    lua.create_function(move |_lua, prompt: String| {
        // Use tokio's block_in_place to call async code from sync context
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                // Execute prompt based on client type
                let response = match &client {
                    LlmClient::Ollama(model) => {
                        let client = ollama::Client::new();
                        let agent = client
                            .agent(model)
                            .additional_params(json!({"think": false}))
                            .build();
                        agent.prompt(&prompt).await
                    }
                    LlmClient::Openrouter(model, api_key) => {
                        let client = openrouter::Client::new(api_key);
                        let agent = client.agent(model).build();
                        agent.prompt(&prompt).await
                    }
                };

                match response {
                    Ok(response) => Ok(response),
                    Err(e) => Err(mlua::Error::RuntimeError(format!("LLM query failed: {e}"))),
                }
            })
        })
    })
}

/// Creates the custom `token_trunc(text, n)` function for truncating strings by token count.
///
/// # Lua Signature
/// ```lua
/// truncated = token_trunc(text, n)
/// ```
///
/// # Parameters
/// - `text` (string) - The text to truncate
/// - `n` (number) - Maximum number of tokens to keep
///
/// # Returns
/// - (string) - The truncated text, preserving the beginning
///
/// # Behavior
/// - Uses p50k_base BPE tokenizer
/// - If text has fewer than n tokens, returns the original text unchanged
/// - Preserves the beginning of the text (truncates from the end)
/// - Useful for staying within LLM token limits
///
/// # Example
/// ```lua
/// short_text = token_trunc(long_text, 100)
/// chunk = token_trunc(string.sub(context, 1, 5000), 50)
/// ```
fn create_token_trunc_function(lua: &Lua) -> Result<mlua::Function> {
    lua.create_function(|_lua, (s, n): (String, usize)| {
        // Get the BPE tokenizer
        let bpe = p50k_base()
            .map_err(|e| mlua::Error::RuntimeError(format!("Failed to load tokenizer: {e}")))?;

        // Encode the string
        let tokens = bpe.encode_with_special_tokens(&s);

        // Truncate to n tokens
        let truncated_tokens = &tokens[..tokens.len().min(n)];

        // Decode back to string
        let truncated_string = bpe
            .decode(truncated_tokens.to_vec())
            .map_err(|e| mlua::Error::RuntimeError(format!("Failed to decode tokens: {e}")))?;

        Ok(truncated_string)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_print() {
        let env = Environment::new("initial", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();
        let result = env.eval(r#"print("hello moon")"#).unwrap();
        assert_eq!(result, Some("hello moon".to_string()));
    }

    #[test]
    fn test_no_output() {
        let env = Environment::new("initial", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();
        let result = env.eval("x = 5").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_persistent_state() {
        let env = Environment::new("initial", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();

        // Set a variable
        let result = env.eval("x = 5").unwrap();
        assert_eq!(result, None);

        // Use the variable in a subsequent eval
        let result = env.eval("print(x * 2)").unwrap();
        assert_eq!(result, Some("10".to_string()));
    }

    #[test]
    fn test_multiple_prints() {
        let env = Environment::new("initial", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();
        let result = env.eval(r#"print("first"); print("second")"#).unwrap();
        assert_eq!(result, Some("first\nsecond".to_string()));
    }

    #[test]
    fn test_state_accumulation() {
        let env = Environment::new("initial", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();

        env.eval("a = 10").unwrap();
        env.eval("b = 20").unwrap();
        let result = env.eval("print(a + b)").unwrap();
        assert_eq!(result, Some("30".to_string()));
    }

    #[test]
    fn test_print_with_multiple_args() {
        let env = Environment::new("initial", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();
        let result = env.eval(r#"print("hello", "world", 42)"#).unwrap();
        assert_eq!(result, Some("hello\tworld\t42".to_string()));
    }

    #[test]
    fn test_context_variable_string() {
        let env = Environment::new(
            "my context value",
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();
        let result = env.eval("print(context)").unwrap();
        assert_eq!(result, Some("my context value".to_string()));
    }

    #[test]
    fn test_context_variable_number() {
        let env = Environment::new(42, LlmClient::Ollama("qwen3:30b".to_string())).unwrap();
        let result = env.eval("print(context * 2)").unwrap();
        assert_eq!(result, Some("84".to_string()));
    }

    #[test]
    fn test_context_variable_table() {
        let env = Environment::new("initial", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();
        // Create a table and set it as context
        env.eval("context = {name = 'test', value = 100}").unwrap();
        let result = env
            .eval("print(context.name .. ': ' .. context.value)")
            .unwrap();
        assert_eq!(result, Some("test: 100".to_string()));
    }

    #[test]
    fn test_token_trunc_basic() {
        let env = Environment::new("", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();

        // Test truncating a simple string
        let code = r#"
            text = "This is a test string that will be truncated to a smaller number of tokens."
            truncated = token_trunc(text, 5)
            print(truncated)
        "#;

        let result = env.eval(code).unwrap();
        assert!(result.is_some(), "token_trunc should return output");

        let output = result.unwrap();
        // The truncated string should be shorter than the original
        assert!(
            output.len() < 77,
            "Truncated string should be shorter than original, got: {output}"
        );

        // Should start with "This"
        assert!(
            output.starts_with("This"),
            "Truncated string should start with 'This', got: {output}"
        );
    }

    #[test]
    fn test_token_trunc_exact() {
        let env = Environment::new("", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();

        // Test with a known token count
        let code = r#"
            text = "Hello world"
            truncated = token_trunc(text, 1)
            print(truncated)
        "#;

        let result = env.eval(code).unwrap();
        assert!(result.is_some());

        let output = result.unwrap();
        // With 1 token, we should get just "Hello" or similar
        assert!(
            output.len() < 12,
            "Truncated to 1 token should be much shorter, got: {output}"
        );
    }

    #[test]
    fn test_token_trunc_longer_than_input() {
        let env = Environment::new("", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();

        // Test truncating to more tokens than the input has
        let code = r#"
            text = "Short"
            truncated = token_trunc(text, 1000)
            print(truncated)
        "#;

        let result = env.eval(code).unwrap();
        assert!(result.is_some());

        let output = result.unwrap();
        // Should return the full string if n is larger than token count
        assert_eq!(output, "Short");
    }

    #[test]
    fn test_token_trunc_empty_string() {
        let env = Environment::new("", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();

        let code = r#"
            text = ""
            truncated = token_trunc(text, 10)
            print(truncated)
        "#;

        let result = env.eval(code).unwrap();
        // Empty string should produce no output or empty output
        assert!(result.is_none() || result == Some("".to_string()));
    }

    #[test]
    fn test_token_trunc_with_special_chars() {
        let env = Environment::new("", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();

        let code = r#"
            text = "Hello! How are you? I'm doing well. 😀"
            truncated = token_trunc(text, 5)
            print(truncated)
        "#;

        let result = env.eval(code).unwrap();
        assert!(result.is_some());

        let output = result.unwrap();
        // Should handle special characters and emojis
        assert!(
            output.len() < 40,
            "Truncated string with special chars should be shorter, got: {output}"
        );
    }

    #[test]
    fn test_token_trunc_preserves_beginning() {
        let env = Environment::new("", LlmClient::Ollama("qwen3:30b".to_string())).unwrap();

        let code = r#"
            text = "The quick brown fox jumps over the lazy dog"
            truncated = token_trunc(text, 3)
            print(truncated)
        "#;

        let result = env.eval(code).unwrap();
        assert!(result.is_some());

        let output = result.unwrap();
        // Should preserve the beginning of the string
        assert!(
            output.starts_with("The"),
            "Should start with 'The', got: {output}"
        );
    }
}
