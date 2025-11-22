use crate::repl::Repl;
use colored::Colorize;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};

#[derive(Deserialize)]
pub struct RunCellArgs {
    pub comment: String,
    pub code: String,
}

#[derive(Clone)]
pub struct RunCellTool {
    repl: Arc<Mutex<Repl>>,
}

impl RunCellTool {
    pub fn new(repl: Arc<Mutex<Repl>>) -> Self {
        Self { repl }
    }
}

#[derive(Debug)]
pub struct RunCellError(String);

impl std::fmt::Display for RunCellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for RunCellError {}

impl Tool for RunCellTool {
    const NAME: &'static str = "run_cell";

    type Error = RunCellError;
    type Args = RunCellArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Execute a Lua code cell in the REPL environment. The code can access and manipulate the 'context' variable, create new global variables, use string operations, regex, etc. Returns the output from print statements or empty string if no output.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "comment": {
                        "type": "string",
                        "description": "A brief comment describing the intent of this code cell"
                    },
                    "code": {
                        "type": "string",
                        "description": "The Lua code to execute in the REPL environment"
                    }
                },
                "required": ["comment", "code"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mut repl = self.repl.lock().unwrap();

        // Print horizontal line if there are already cells
        if !repl.entries.is_empty() {
            println!();
            println!("{}", "─".repeat(80));
            println!();
        }

        // Print comment in bold
        println!("{}", args.comment.bold());

        // Space
        println!();

        // Print code in regular text color
        println!("{}", args.code);

        // Space
        println!();

        // Call the Repl's eval method
        repl.eval(&args.comment, &args.code);

        // Get the output from the last entry
        let output = repl.entries.last().and_then(|cell| cell.output.clone());

        // Print output in bold with arrow prefix
        let output_display = match &output {
            None => format!("→ {}", "(no output)"),
            Some(out) => format!("→ {out}"),
        };
        println!("{}", output_display.bold());

        Ok(output.unwrap_or_default())
    }
}
