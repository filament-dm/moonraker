use crate::environment::{Environment, LlmClient};
use crate::rlm::{LmInput, OutputParser};
use mlua::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::error::Error;
use tiktoken_rs::p50k_base;

/// Maximum tokens allowed for cell output in context
const MAX_OUTPUT_TOKENS: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Cell {
    /// Description of the intent of this cell.
    pub comment: String,

    /// Computation to perform.
    pub code: String,

    /// Output of computation. Partial cells have this set to None.
    pub output: Option<String>,

    /// True if this is the final cell in the computation sequence.
    #[serde(default)]
    pub r#final: bool,
}

impl OutputParser for Cell {
    fn parse(text: &str) -> std::result::Result<Self, Box<dyn Error>> {
        use regex::Regex;

        // Try to parse as JSON first for backward compatibility
        if let Ok(cell) = serde_json::from_str::<Cell>(text) {
            return Ok(cell);
        }

        // Parse using XML tags with regex (using (?s) for multiline matching)
        let comment_re = Regex::new(r"(?s)<comment>(.*?)</comment>").unwrap();
        let code_re = Regex::new(r"(?s)<code>(.*?)</code>").unwrap();
        let final_re = Regex::new(r"(?s)<final>(.*?)</final>").unwrap();

        // Extract comment
        let comment = comment_re
            .captures(text)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string())
            .ok_or("Failed to parse <comment> tag from response")?;

        // Extract code
        let code = code_re
            .captures(text)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string())
            .ok_or("Failed to parse <code> tag from response")?;

        // Extract final flag (optional)
        let final_flag = final_re
            .captures(text)
            .and_then(|cap| cap.get(1))
            .map(|m| {
                let value = m.as_str().trim().to_lowercase();
                value == "true" || value == "yes"
            })
            .unwrap_or(false);

        // Validate that we got comment and code
        if comment.is_empty() {
            return Err("Comment tag is empty".into());
        }
        if code.is_empty() {
            return Err("Code tag is empty".into());
        }

        Ok(Cell {
            comment,
            code,
            output: None,
            r#final: final_flag,
        })
    }
}

pub struct Repl {
    pub prompt: String,
    pub entries: Vec<Cell>,
    environment: Environment,
}

impl Serialize for Repl {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Repl", 2)?;
        state.serialize_field("prompt", &self.prompt)?;
        state.serialize_field("entries", &self.entries)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Repl {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ReplData {
            prompt: String,
            entries: Vec<Cell>,
        }

        let data = ReplData::deserialize(deserializer)?;

        // Create a new environment with a default context when deserializing
        let environment = Environment::new("", LlmClient::Ollama("qwen3:30b".to_string()))
            .map_err(serde::de::Error::custom)?;

        Ok(Repl {
            prompt: data.prompt,
            entries: data.entries,
            environment,
        })
    }
}

impl Repl {
    pub fn new<T>(
        prompt: String,
        init_context: T,
        _model: String,
        client: LlmClient,
    ) -> Result<Self>
    where
        T: mlua::IntoLua,
    {
        Ok(Repl {
            prompt,
            entries: Vec::new(),
            environment: Environment::new(init_context, client)?,
        })
    }

    pub fn eval(&mut self, comment: &str, code: &str) {
        let output = match self.environment.eval(code) {
            Ok(Some(result)) => {
                // Truncate output to MAX_OUTPUT_TOKENS
                if let Ok(bpe) = p50k_base() {
                    let tokens = bpe.encode_with_special_tokens(&result);
                    if tokens.len() > MAX_OUTPUT_TOKENS {
                        let truncated_tokens = &tokens[..MAX_OUTPUT_TOKENS];
                        if let Ok(decoded) = bpe.decode(truncated_tokens.to_vec()) {
                            Some(format!("{decoded}\n[truncated]"))
                        } else {
                            Some(result)
                        }
                    } else {
                        Some(result)
                    }
                } else {
                    Some(result)
                }
            }
            Ok(None) => None,
            Err(e) => Some(format!("Execution error: {e}")),
        };

        self.entries.push(Cell {
            comment: comment.to_string(),
            code: code.to_string(),
            output,
            r#final: false,
        });
    }

    /// Create a snapshot of the REPL state (prompt and entries) without the environment
    /// Used for serialization and passing to LMs
    pub fn snapshot(&self) -> Result<Self> {
        Ok(Repl {
            prompt: self.prompt.clone(),
            entries: self.entries.clone(),
            environment: Environment::new("", LlmClient::Ollama("qwen3:30b".to_string()))?,
        })
    }

    pub fn to_markdown(&self) -> String {
        let mut parts = Vec::new();

        // Add the prompt if it exists
        if !self.prompt.is_empty() {
            parts.push(format!("Prompt:\n{}\n", self.prompt));
        }

        // Format each cell
        for cell in &self.entries {
            let mut cell_parts = Vec::new();

            // Add comment as markdown heading
            if !cell.comment.is_empty() {
                cell_parts.push(format!("# {}", cell.comment));
            }

            // Add code in triple backticks
            if !cell.code.is_empty() {
                cell_parts.push(format!("```\n{}\n```", cell.code));
            }

            // Add output in triple backticks if it exists (already truncated in eval)
            if let Some(output) = &cell.output {
                cell_parts.push(format!("Output:\n```\n{output}\n```"));
            }

            // Join cell parts and add to main parts
            if !cell_parts.is_empty() {
                parts.push(format!("{}\n", cell_parts.join("\n")));
            }
        }

        parts.join("\n")
    }
}

impl LmInput for Repl {
    fn format(&self) -> String {
        self.to_markdown()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_basic_eval() {
        let mut repl = Repl::new(
            "test prompt".to_string(),
            "test",
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();
        repl.eval("Print hello", r#"print("hello")"#);

        assert_eq!(repl.entries.len(), 1);
        assert_eq!(repl.entries[0].comment, "Print hello");
        assert_eq!(repl.entries[0].code, r#"print("hello")"#);
        assert_eq!(repl.entries[0].output, Some("hello".to_string()));
    }

    #[test]
    fn test_repl_no_output() {
        let mut repl = Repl::new(
            "test prompt".to_string(),
            "test",
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();
        repl.eval("Set variable", "x = 5");

        assert_eq!(repl.entries.len(), 1);
        assert_eq!(repl.entries[0].comment, "Set variable");
        assert_eq!(repl.entries[0].code, "x = 5");
        assert_eq!(repl.entries[0].output, None);
    }

    #[test]
    fn test_repl_persistent_state() {
        let mut repl = Repl::new(
            "test prompt".to_string(),
            "test",
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();

        repl.eval("Set x", "x = 5");
        repl.eval("Print x * 2", "print(x * 2)");

        assert_eq!(repl.entries.len(), 2);
        assert_eq!(repl.entries[0].output, None);
        assert_eq!(repl.entries[1].output, Some("10".to_string()));
    }

    #[test]
    fn test_repl_error_handling() {
        let mut repl = Repl::new(
            "test prompt".to_string(),
            "test",
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();
        repl.eval("Invalid code", "this is not valid lua");

        assert_eq!(repl.entries.len(), 1);
        assert!(
            repl.entries[0]
                .output
                .as_ref()
                .unwrap()
                .starts_with("Execution error:")
        );
    }

    #[test]
    fn test_repl_serialization() {
        let mut repl = Repl::new(
            "test prompt".to_string(),
            "test",
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();
        repl.eval("First cell", r#"print("output1")"#);
        repl.eval("Second cell", "x = 10");

        let json = serde_json::to_string(&repl).unwrap();
        assert!(json.contains("test prompt"));
        assert!(json.contains("First cell"));
        assert!(json.contains("Second cell"));
        assert!(json.contains("output1"));
    }

    #[test]
    fn test_repl_deserialization() {
        let json = r#"{
            "prompt": "deserialization test prompt",
            "entries": [
                {
                    "comment": "Test comment",
                    "code": "print('hello')",
                    "output": "hello"
                }
            ]
        }"#;

        let repl: Repl = serde_json::from_str(json).unwrap();
        assert_eq!(repl.prompt, "deserialization test prompt");
        assert_eq!(repl.entries.len(), 1);
        assert_eq!(repl.entries[0].comment, "Test comment");
        assert_eq!(repl.entries[0].code, "print('hello')");
        assert_eq!(repl.entries[0].output, Some("hello".to_string()));
    }

    #[test]
    fn test_repl_context_access() {
        let mut repl = Repl::new(
            "test prompt".to_string(),
            "my context",
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();
        repl.eval("Print context", "print(context)");

        assert_eq!(repl.entries[0].output, Some("my context".to_string()));
    }

    #[test]
    fn test_repl_multiple_evals() {
        let mut repl = Repl::new(
            "test prompt".to_string(),
            0,
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();

        repl.eval("First", "a = 1");
        repl.eval("Second", "b = 2");
        repl.eval("Third", "c = 3");
        repl.eval("Sum", "print(a + b + c)");

        assert_eq!(repl.entries.len(), 4);
        assert_eq!(repl.entries[3].output, Some("6".to_string()));
    }

    #[test]
    fn test_repl_lm_input_format() {
        let mut repl = Repl::new(
            "This is the main prompt".to_string(),
            0,
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();

        repl.eval("Set variable x", "x = 10");
        repl.eval("Calculate result", "print(x * 2)");

        let formatted = repl.format();

        // Check that prompt is included
        assert!(formatted.contains("This is the main prompt"));

        // Check that comments are formatted as markdown headings
        assert!(formatted.contains("# Set variable x"));
        assert!(formatted.contains("# Calculate result"));

        // Check that code is in triple backticks
        assert!(formatted.contains("```\nx = 10\n```"));
        assert!(formatted.contains("```\nprint(x * 2)\n```"));

        // Check that output is in triple backticks
        assert!(formatted.contains("```\n20\n```"));
    }

    #[test]
    fn test_repl_lm_input_format_no_output() {
        let mut repl = Repl::new(
            "test prompt".to_string(),
            0,
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();

        repl.eval("Set variable", "y = 5");

        let formatted = repl.format();

        // Should have comment and code, but no output section
        assert!(formatted.contains("# Set variable"));
        assert!(formatted.contains("```\ny = 5\n```"));

        // Should only have two code blocks (one for code), not three
        assert_eq!(formatted.matches("```").count(), 2);
    }

    #[test]
    fn test_cell_parser_xml_format() {
        let text = r#"<comment>
First, let me check the context to understand what we're working with
</comment>

<code>
print(string.sub(context, 1, 100))
</code>

<final>
false
</final>"#;

        let cell = Cell::parse(text).unwrap();
        assert_eq!(
            cell.comment,
            "First, let me check the context to understand what we're working with"
        );
        assert_eq!(cell.code, "print(string.sub(context, 1, 100))");
        assert!(!cell.r#final);
    }

    #[test]
    fn test_cell_parser_final_true() {
        let text = r#"<comment>
Output the final answer
</comment>

<code>
print("The answer is: 42")
</code>

<final>
true
</final>"#;

        let cell = Cell::parse(text).unwrap();
        assert_eq!(cell.comment, "Output the final answer");
        assert_eq!(cell.code, r#"print("The answer is: 42")"#);
        assert!(cell.r#final);
    }

    #[test]
    fn test_cell_parser_json_fallback() {
        let json = r#"{"comment": "Test comment", "code": "print('hello')", "final": false}"#;
        let cell = Cell::parse(json).unwrap();
        assert_eq!(cell.comment, "Test comment");
        assert_eq!(cell.code, "print('hello')");
        assert!(!cell.r#final);
    }

    #[test]
    fn test_repl_lm_input_format_example() {
        let mut repl = Repl::new(
            "Calculate fibonacci numbers".to_string(),
            0,
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();

        repl.eval(
            "Define fibonacci function",
            r#"function fib(n)
  if n <= 1 then
    return n
  else
    return fib(n-1) + fib(n-2)
  end
end"#,
        );

        repl.eval("Calculate fib(5)", "print(fib(5))");
        repl.eval("Calculate fib(10)", "print(fib(10))");

        let formatted = repl.format();

        // Print for visual inspection
        println!("\n=== Formatted Repl Output ===\n{formatted}\n=========================\n");

        // Basic assertions
        assert!(formatted.contains("Calculate fibonacci numbers"));
        assert!(formatted.contains("# Define fibonacci function"));
        assert!(formatted.contains("# Calculate fib(5)"));
        assert!(formatted.contains("```\n5\n```"));
        assert!(formatted.contains("```\n55\n```"));
    }

    #[test]
    fn test_output_truncation() {
        let mut repl = Repl::new(
            "Test truncation".to_string(),
            0,
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();

        // Create a very long output that will exceed 100 tokens
        let long_output_code = r#"
            output = ""
            for i = 1, 200 do
                output = output .. "This is line " .. i .. " with some additional text to make it longer. "
            end
            print(output)
        "#;

        repl.eval("Generate long output", long_output_code);

        // Get the formatted markdown
        let formatted = repl.format();

        // The output should be truncated and contain [truncated]
        assert!(
            formatted.contains("[truncated]"),
            "Long output should be truncated and contain [truncated] marker"
        );

        // The cell's output should also contain [truncated] since we truncate at storage time
        assert!(repl.entries[0].output.is_some());
        let cell_output = repl.entries[0].output.as_ref().unwrap();
        assert!(
            cell_output.contains("[truncated]"),
            "Cell output should contain [truncated] marker"
        );

        // The truncated output should be much shorter than the original would have been
        assert!(
            cell_output.len() < 1000,
            "Truncated output should be much shorter than original"
        );
    }

    #[test]
    fn test_short_output_not_truncated() {
        let mut repl = Repl::new(
            "Test no truncation".to_string(),
            0,
            "test-model".to_string(),
            LlmClient::Ollama("qwen3:30b".to_string()),
        )
        .unwrap();

        repl.eval("Short output", r#"print("Hello world")"#);

        let formatted = repl.format();

        // Short output should not be truncated
        assert!(
            !formatted.contains("[truncated]"),
            "Short output should not be truncated"
        );
        assert!(formatted.contains("Hello world"));
    }
}
