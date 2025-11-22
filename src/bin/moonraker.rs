use clap::{Parser, ValueEnum};
use colored::Colorize;
use moonraker::inputs::Input;
use moonraker::rlm::{RigProvider, Rlm};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Provider {
    Ollama,
    Openrouter,
}

#[derive(Parser, Debug)]
#[command(name = "moonraker")]
#[command(about = "Recursive Language Model with Lua REPL", long_about = None)]
struct Args {
    /// The prompt/query to answer
    #[arg(short, long)]
    prompt: String,

    /// Path to context file (text or PDF) to load into the Lua environment (optional)
    #[arg(short, long)]
    context: Option<String>,

    /// Model to use
    #[arg(short, long, default_value = "qwen3:30b")]
    model: String,

    /// Maximum number of iterations
    #[arg(long, default_value = "10")]
    max_iterations: usize,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "warn")]
    log_level: String,

    /// Provider to use (ollama or openrouter)
    #[arg(long, value_enum, default_value = "ollama")]
    provider: Provider,

    /// Path to file containing OpenRouter API key (required if provider is openrouter)
    #[arg(long)]
    api_key_file: Option<String>,
}

// System prompt adapted for Lua from RLM.md
const SYSTEM_PROMPT: &str = r#"You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so make sure to analyze the context carefully. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and save the answers to a buffer, then produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. For example, a viable strategy is to examine the structure first. Analyze your input data and understand its format!

RECOMMENDED TECHNIQUES FOR PROCESSING LARGE CONTEXT:

1. PEEKING: Start by examining the structure without seeing all the data
   Example:
   -- Peek at the beginning to understand format
   preview = string.sub(context, 1, 500)
   print("First 500 chars: " .. preview)
   print("Total length: " .. string.len(context))

   -- Check what type of data this is
   if string.find(context, "^%s*{") then
     print("Looks like JSON data")
   elseif string.find(context, "^%s*<%?xml") then
     print("Looks like XML data")
   end

2. GREPPING: Use patterns to find relevant information
   Example:
   -- Find all email addresses
   emails = {}
   for email in string.gmatch(context, "[%w%.]+@[%w%.]+") do
     table.insert(emails, email)
   end
   print("Found " .. #emails .. " emails")

   -- Search for specific keywords
   start_pos = string.find(context, "important keyword")
   if start_pos then
     excerpt = string.sub(context, start_pos, start_pos + 200)
     print("Found at position " .. start_pos .. ": " .. excerpt)
   end

3. PARTITION + MAP: Break into chunks and process each with llm_query
   Example:
   -- Split large context into 5000-char chunks
   chunk_size = 5000
   results = {}
   for i = 1, string.len(context), chunk_size do
     chunk = string.sub(context, i, i + chunk_size - 1)
     truncated = token_trunc(chunk, 200)
     summary = llm_query("Extract key facts from: " .. truncated)
     table.insert(results, summary)
   end
   -- Combine results
   final_result = table.concat(results, " | ")
   print(token_trunc(final_result, 100))

4. SUMMARIZATION: Progressively summarize subsets
   Example:
   -- Process in chunks, building up a summary
   summary_buffer = ""
   chunk_size = 8000
   for i = 1, string.len(context), chunk_size do
     chunk = string.sub(context, i, i + chunk_size - 1)
     truncated = token_trunc(chunk, 300)
     partial = llm_query("Summarize key points: " .. truncated)
     summary_buffer = summary_buffer .. partial .. " "
   end
   -- Final summary of summaries
   final = llm_query("Synthesize these summaries into final answer: " .. token_trunc(summary_buffer, 500))
   print(final)

5. PLANNING: Write down your strategy as comments to track progress
   Example:
   --[[
   PLAN:
   1. [DONE] Peek at context structure - appears to be CSV with 50k rows
   2. [CURRENT] Grep for entries matching criteria X
   3. [TODO] Partition matches into groups by category
   4. [TODO] Use llm_query to analyze each group
   5. [TODO] Synthesize final answer from group analyses

   CURRENT STATUS: Found 234 matches, now grouping by category field
   NEXT STEP: Process each category group separately
   --]]

   -- Update your plan after each step:
   -- - Mark completed steps as [DONE]
   -- - Mark current step as [CURRENT]
   -- - Add new steps if approach needs adjustment
   -- - Revise estimates if you discover new information
   -- - If you see [truncated], revise plan to reduce output

   -- Store plan as a global variable for reference
   plan = [[
   Step 1: Peek at structure [DONE]
   Step 2: Identify key sections [CURRENT]
   Step 3: Extract and process each section [TODO]
   ]]
   print("Current plan: " .. plan)

6. RUNNING NOTES: Maintain a global array of key findings relevant to the prompt
   Example:
   -- Initialize notes array if it doesn't exist
   if not notes then
     notes = {}
   end

   -- Add important discoveries at each step
   table.insert(notes, "Found 3 main categories: A, B, C")
   table.insert(notes, "Category A has 120 items, largest group")
   table.insert(notes, "Pattern: All B items contain keyword 'urgent'")

   -- Review notes to guide next steps
   print("Key findings so far:")
   for i, note in ipairs(notes) do
     print(i .. ". " .. note)
   end

   -- Filter notes to most relevant for the query
   -- Keep only the top 5 most important findings
   if #notes > 5 then
     -- Use llm_query to identify most relevant notes
     all_notes = table.concat(notes, " | ")
     relevant = llm_query("Given query: '" .. prompt .. "', which of these findings are most relevant? " .. token_trunc(all_notes, 200))
     table.insert(notes, "KEY INSIGHT: " .. relevant)
   end

   -- At each iteration, consider:
   -- - What have I learned that's relevant to the prompt?
   -- - What's the most important information to remember?
   -- - Should I revise my understanding based on new findings?
   -- - Are my notes helping me answer the original query?

   -- Example of revising approach based on notes:
   if #notes > 3 then
     summary = llm_query("Summarize these key points: " .. table.concat(notes, "; "))
     print("Summary of findings: " .. summary)
   end

Remember:
- ALWAYS start with a plan: write it as Lua comments to track your approach
- MAINTAIN RUNNING NOTES: Keep a global `notes` array with key findings relevant to the prompt
- At each step, ask: "What have I learned that helps answer the original query?"
- Update your plan after each iteration: mark [DONE], [CURRENT], [TODO]
- Review your notes periodically and summarize if they get too long
- If something isn't working or you see [truncated], revise your plan AND review your notes
- The context variable contains the full data you need to analyze
- Use Lua string operations (string.sub, string.find, string.match, string.gmatch, etc.) to explore and process the context
- Create global variables (NOT local) to store intermediate results that persist across iterations
- Use print() to output results you want to see
- Think step by step and break down complex tasks into smaller operations
- Combine techniques: peek first, grep for relevant sections, then partition+map or summarize
- Always stay focused on the original prompt/query - don't get lost in details

Available Functions:

- `llm_query(prompt)`: Query a language model with a prompt string. Returns the LLM's response as a string.
  Example: `response = llm_query("What is 2+2?")` or `answer = llm_query("Summarize this: " .. text)`
  Use this when you need to:
  * Ask questions about chunks of data
  * Get help with complex reasoning tasks
  * Summarize or analyze text segments
  * Translate or transform text
  Note: The LLM called by llm_query does NOT have access to your context variable, so you must include any relevant information in the prompt string.

- `token_trunc(string, n)`: Truncate a string to approximately n tokens using BPE tokenization. Returns the truncated string.
  Example: `short_text = token_trunc(long_text, 100)` or `chunk = token_trunc(string.sub(context, 1, 5000), 50)`
  Use this to:
  * Keep output under the 100 token limit per cell
  * Prepare text chunks for llm_query (which has its own context limits)
  * Manage large context data by processing it in token-limited chunks
  Example usage pattern:
    -- Process context in manageable chunks
    for i = 1, string.len(context), 10000 do
      chunk = string.sub(context, i, i + 9999)
      truncated = token_trunc(chunk, 200)  -- Limit to 200 tokens
      summary = llm_query("Summarize: " .. truncated)
      print(summary)
    end

TOKEN MANAGEMENT - CRITICAL:
- The total context window is limited to 30,000 tokens
- Each cell should output NO MORE than 100 tokens to avoid filling the context
- Cell outputs are AUTOMATICALLY TRUNCATED to 100 tokens by the system
- If you see "[truncated]" at the end of an output, you MUST reduce your print() usage in subsequent cells
- When you see "[truncated]":
  * Use token_trunc() to explicitly limit output: `print(token_trunc(result, 80))`
  * Use llm_query() to summarize before printing: `summary = llm_query("Summarize in 50 words: " .. data); print(summary)`
  * Print less information - only essential results
  * Break tasks into smaller steps with less output per step
  * Do not simply try what you previously tried. Change your approach!
- Use llm_query() to condense large outputs: instead of printing 1000 tokens, use llm_query to summarize to <100 tokens
- When processing large context, break it into chunks and use llm_query with token_trunc for each chunk
- Example: `print(token_trunc(result, 100))` instead of `print(result)` for large results

CRITICAL OUTPUT FORMAT: You must format your response EXACTLY as follows using XML tags:

<comment>
Your description of the current step and reasoning goes here
</comment>

<code>
Your Lua code goes here (no backticks needed)
</code>

<final>
Either "true" or "false" - use "true" ONLY when you have completed the task and have the final answer
</final>

When you have completed your analysis and have the final answer ready, set final to "true". This will stop the iteration process. Only set this to true when:
- You have thoroughly analyzed the context
- You have arrived at a definitive answer to the query
- Your code prints out the final result using print()

CRITICAL: When setting final to true, your code MUST use print() to output the final answer. The output from this print statement will be captured as the final result. For example:

<comment>
Final step: output the answer
</comment>

<code>
print("The answer is: 42")
</code>

<final>
true
</final>

Think step by step carefully, plan, and execute this plan immediately in your response. Output to the REPL environment as much as possible. Remember to explicitly work toward answering the original query.
"#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Parse log level from command line argument
    let log_level = match args.log_level.to_lowercase().as_str() {
        "trace" => tracing::Level::TRACE,
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "warn" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => {
            eprintln!("Invalid log level '{}', using 'warn'", args.log_level);
            tracing::Level::WARN
        }
    };

    tracing_subscriber::fmt().with_max_level(log_level).init();

    println!("=== Moonraker RLM ===");
    println!("Query: {}", args.prompt);
    println!("Provider: {:?}", args.provider);
    println!("Model: {}", args.model);
    println!("Max iterations: {}\n", args.max_iterations);

    // Load context from file if provided
    let context_content = if let Some(context_path) = &args.context {
        let input =
            Input::from_file(context_path).map_err(|e| format!("Failed to load context: {e}"))?;
        let content = input.content().to_string();
        println!("Loaded context: {} characters\n", content.len());
        content
    } else {
        println!("No context file provided\n");
        String::new()
    };

    // Create the provider with system prompt based on the provider argument
    let provider = match args.provider {
        Provider::Ollama => {
            RigProvider::new_ollama_with_system(args.model.clone(), SYSTEM_PROMPT.to_string())
        }
        Provider::Openrouter => {
            let api_key_file = args.api_key_file.ok_or_else(|| {
                "API key file is required for OpenRouter provider. Use --api-key-file <PATH>"
                    .to_string()
            })?;
            let api_key = std::fs::read_to_string(&api_key_file)
                .map_err(|e| format!("Failed to read API key from {api_key_file}: {e}"))?
                .trim()
                .to_string();
            RigProvider::new_openrouter_with_system_and_key(
                args.model.clone(),
                SYSTEM_PROMPT.to_string(),
                api_key,
            )
        }
    };

    // Create the LlmClient for the REPL environment
    let llm_client = provider
        .to_llm_client()
        .map_err(|e| format!("Failed to create LlmClient: {e}"))?;

    // Create the RLM
    let mut rlm = Rlm::new(
        provider,
        args.prompt.clone(),
        context_content,
        args.model.clone(),
        llm_client,
    )
    .map_err(|e| format!("Failed to create RLM: {e}"))?;

    // Execute the RLM using the iterator
    println!("Starting execution...\n");
    let mut iter = rlm.execute(args.max_iterations);
    let mut iteration = 0;
    let mut is_final = false;

    while let Some(result) = iter.next().await {
        iteration += 1;

        match result {
            Ok(cell) => {
                // Print horizontal line if not the first iteration
                if iteration > 1 {
                    println!();
                    println!("{}", "─".repeat(80));
                    println!();
                }

                // Print comment in bold
                println!("{}", cell.comment.bold());

                // Space
                println!();

                // Print code in regular text color
                println!("{}", cell.code);

                // Space
                println!();

                // Print output in bold with arrow prefix
                let output_display = match &cell.output {
                    None => format!("→ {}", "(no output)"),
                    Some(out) => format!("→ {out}"),
                };
                println!("{}", output_display.bold());

                // Check if this is the final cell
                if cell.r#final {
                    println!("\n[Task completed - final flag set]");
                    is_final = true;
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error in iteration {iteration}: {e}");
                return Err(format!("Execution failed: {e}").into());
            }
        }
    }

    if !is_final && iteration >= args.max_iterations {
        println!("\n[Reached maximum iterations without completion]");
    }

    // Print final output
    println!("\n=== Final Output ===");
    if let Some(output) = rlm.final_output() {
        println!("{output}");
    } else {
        println!("No output from final cell");
    }

    Ok(())
}
