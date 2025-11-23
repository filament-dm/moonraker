# Moonraker

A Lua REPL agent system for processing and analyzing large contexts.

## Overview

Moonraker implements a novel approach to handling large contexts by giving an LLM agent access to a persistent Lua REPL environment. Instead of trying to fit massive amounts of data into the model's context window, the agent can write Lua code to explore, analyze, and manipulate the data programmatically.

This project is inspired by Alex Zhang's work on **Reinforcement Learning over In-Context RL Machines (RLM)**[^1]. The core insight is that by providing an agent with computational tools (in this case, a Lua REPL), it can break down complex analysis tasks into smaller, manageable steps—much like a data scientist working in a Jupyter notebook.

## How It Works

1. **Context Storage**: Large amounts of data are stored in a Lua global variable called `context`
2. **Agent Loop**: An LLM agent receives your prompt and can call the `run_cell` tool
3. **Iterative Analysis**: The agent writes Lua code cells to explore and analyze the context
4. **State Persistence**: All global variables and state persist across cell executions
5. **Final Answer**: After building up its analysis, the agent provides a final answer

### Hypothetical Session

This illustrates the concept for clarity but is not an output from an actual run.

```
User Prompt: "Find lines containing 'ERROR' in the logs"
Context: [Large log file content]

Agent thinks:
  → Cell 1: Print first 100 chars to understand format
  → Cell 2: Use string.gmatch to find all lines with "ERROR"
  → Cell 3: Count the error lines
  → Cell 4: Extract unique error messages

Agent responds: "Found 42 ERROR lines with 5 unique error types..."
```

[EXAMPLE.txt] contains an actual Qwen3 235B A22B run.

## Installation

### Prerequisites

1. Rust
2. Ollama or an OpenRouter account

Development and testing is done with the Ollama hosted qwen3:30b model, which in Ollama's default 4 bit quant form will require about 32GB of GPU or unified memory. In production you'll likely need a smarter model provided by OpenRouter (Qwen3 235B A22B works reasonably well, as does GPT-5 Mini).

### Building

```bash
cargo build
```

### Testing

Run unit tests (no external dependencies):
```bash
cargo test
```

Run integration tests (requires Ollama and qwen3:30b):
```bash
# Ensure Ollama is running and model is pulled:
# ollama pull qwen3:30b
cargo test --features integration
```

## Usage

### Basic

```bash
cargo run -- --prompt "Your question here" --context path/to/file.txt [--provider ollama] [--model qwen3:30b]
```

We support `ollama` and `openrouter` providers and their model identifiers, e.g:

```bash
# Using Ollama (default)
cargo run -- --prompt "Your question" --context file.txt

# Using OpenRouter (requires API key file)
cargo run -- --prompt "Your question" --context file.txt --provider openrouter --model openai/gpt-4o --api-key-file openrouter.key
```

### Supported Context File Types

Moonraker can automatically load context from:

- **Text files** (`.txt`, `.log`, `.json`, `.csv`, etc.) - Any UTF-8 text file
- **PDF files** (`.pdf`) - Automatically extracts text content using `lopdf`

### Examples

#### Example 1: Analyze Text Files

```bash
echo "The quick brown fox jumps over the lazy dog" > sample.txt
cargo run -- \
  --prompt "Count how many times the letter 'e' appears in the context" \
  --context sample.txt
```

#### Example 2: Analyze Log Files

```bash
cargo run -- \
  --prompt "Find all ERROR lines and count them by error type" \
  --context application.log
```

#### Example 3: Extract Data from Large Files

```bash
cargo run -- \
  --prompt "Extract all email addresses and list them" \
  --context large_document.txt
```

#### Example 4: Process JSON Files

```bash
cat > users.json << 'EOF'
{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
EOF

cargo run -- \
  --prompt "Calculate the average age of all users" \
  --context users.json
```

#### Example 5: Analyze PDF Documents

```bash
cargo run -- \
  --prompt "Summarize the main topics discussed in this document" \
  --context research_paper.pdf
```

#### Example 6: Math

```bash
cargo run -- \
  --prompt "calculate the nth term of the fibonacci sequence where n is found in the context" \
  --context input.txt
```

## Lua Capabilities

Lua excels at sandboxing and is a good target language for LLMs as it's possible to disable many runtime functions of the Lua interpreter and to replace its built-ins with safe calls. See [src/environment.rs] for further details.

## Testing

### Datasets

`tests/data/` contains some public domain Project Gutenberg texts, along with a unique LLM generated scifi story. These are useful for testing functionality:

### Rig Structured Output Tests

The test suite includes several tests marked with `#[should_panic]` that verify Rig's structured output capabilities with Ollama. These tests currently fail because of a known issue in `llama.cpp` (which Ollama uses under the hood) with JSON schema validation. We want to use structured output in the future for Ollama, so these tests are present to check when the fixed functionality becomes available.

## Future Work

- Support for more Rig providers (OpenAI, Anthropic, etc.)
- Recursive RLM implementation for `llm_query`
- Migrate to Rig structured output once llama.cpp JSON schema support is fixed
- Truncation seems to adversely affect performance. We should figure out a smarter way to budget token usage.

## License

GPLv3.

---

**Moonraker** - Because sometimes you need to reach for the moon to analyze your data 🌙✨

[//]: # "Footnotes"

[^1]: Zhang, Alex and Khattab, Omar. "Recursive Language Models." October 2025. [https://alexzhang13.github.io/blog/2025/rlm/](https://alexzhang13.github.io/blog/2025/rlm/)
