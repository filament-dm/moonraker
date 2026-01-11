# MR-RLM: Map-Reduce Recursive Language Model

## Implementation Specification

### Overview

MR-RLM is a command-line tool that answers natural language prompts over arbitrarily large document sets. It works by recursively applying `flat_map` and `reduce` operations to compress data until it fits within an LLM's context window, then answering directly.

This is a constrained variant of the Recursive Language Model (Zhang et al., 2025) that restricts the programming space to functional map/reduce operations.

### Core Algorithm

```
invoke(docs, prompt, log=[]) → Answer(text) | NotSure(reason)

  # Base case: data fits in context
  if tokens(docs, prompt) < CONTEXT_LIMIT:
    return llm_answer(docs, prompt)
  
  # Recursive case: reduce data and recurse
  for attempt in 1..MAX_ATTEMPTS:
    
    # LLM writes a pipeline given the prompt, doc metadata, and failure log
    pipeline = llm_write_pipeline(docs, prompt, log)
    
    # Execute pipeline
    new_docs = pipeline.reduce_fn(
      flat_map(docs, pipeline.flat_map_fn)
    )
    
    # Recurse with fresh log
    result = invoke(new_docs, prompt, log=[])
    
    if result.type == "answer":
      return result
    
    # Record failure and retry
    log.append({
      pipeline: pipeline,
      input_size: len(docs),
      output_size: len(new_docs),
      failure_reason: result.reason
    })
  
  return NotSure("Max attempts reached. Last failure: " + log[-1].reason)
```

### Data Types

#### Item

An `Item` is either a `Document` (original source text) or a `Thought` (LLM-generated content).

```lua
Document = {
  type = "document",
  content = string,
  source = {
    file = string,   -- original filename
    start = int,     -- byte offset start
    ["end"] = int    -- byte offset end
  }
}

Thought = {
  type = "thought",
  value = any   -- schema-free, whatever the LLM returns
}
```

#### Result

The system returns one of two result types:

```
Answer(text: string)
  -- The LLM successfully answered the prompt
  
NotSure(reason: string)  
  -- The LLM could not answer and explains why
  -- e.g., "No documents mention the requested topic"
  -- e.g., "Found conflicting information"
```

### Pipeline Structure

Each iteration, the LLM produces two Lua functions:

#### flat_map_fn(item) → list[item]

Transforms each item. The return value determines the outcome:

- Return `{}` to drop the item (filtering)
- Return `{item}` to keep unchanged
- Return `{a, b, c, ...}` to expand into multiple items (e.g., chunking)
- Return `{transformed_item}` to transform

Examples:

```lua
-- Filter: keep only items matching a pattern
function(item)
  if fuzzy_match(item, "Q3.*revenue") then
    return {item}
  else
    return {}
  end
end

-- Chunk: split large documents
function(item)
  return chunk(item, 4000)
end

-- Transform: summarize each item via recursive invocation
function(item)
  return {invoke(item, "Summarize the key points")}
end

-- Filter + Transform combined
function(item)
  if semantic_score(item, "financial results") > 0.3 then
    return {invoke(item, "Extract financial figures")}
  else
    return {}
  end
end
```

#### reduce_fn(items) → items

Aggregates the flat_mapped results. Common patterns:

```lua
-- Identity (no reduction)
function(items) return items end

-- Take top N
function(items) return take(items, 50) end

-- Rank by relevance then take top N  
function(items)
  return take(semantic_rank(items, "quarterly revenue"), 20)
end

-- Combine into single item via recursive invocation
function(items)
  return {invoke(items, "Synthesize these into a single summary")}
end
```

### Built-in Functions

The Lua sandbox exposes only these functions:

#### Chunking

```lua
chunk(item, size) → list[item]
```
Split an item into chunks of approximately `size` tokens. Child chunks inherit the source file reference with updated byte offsets.

#### Text Matching

```lua
fuzzy_match(item, pattern) → bool
```
Fuzzy regex match against item content. Should tolerate minor typos, case differences, and flexible whitespace. The pattern language is regex-like but forgiving.

```lua
exact_match(item, pattern) → bool
```
Literal substring match (case-insensitive).

#### Semantic Operations

```lua
semantic_score(item, query) → float
```
Returns embedding similarity between item content and query string. Value ranges from 0.0 to 1.0.

```lua
semantic_rank(items, query) → items
```
Returns items sorted by descending semantic similarity to query.

#### Recursive Invocation

```lua
invoke(items, prompt) → Thought
```
Recursively invokes the entire MR-RLM system on a subset of data. The `items` parameter can be a single item or a list. Returns a `Thought` containing whatever the sub-invocation produced.

This is how the LLM can summarize chunks, extract information, or perform any complex operation that requires LLM reasoning.

#### Utilities

```lua
content(item) → string
```
Extract the text content from an item.

```lua
tokens(item) → int
tokens(items) → int
```
Count tokens in an item or list of items.

```lua
len(items) → int
```
Count number of items in a list.

```lua
take(items, n) → items  
```
Return the first `n` items.

```lua
concat(a, b, ...) → items
```
Concatenate multiple lists into one.

### LLM Prompts

#### Base Case Prompt (data fits in context)

```
You have access to the following documents:

<documents>
{serialized documents with source references}
</documents>

Answer this prompt: {prompt}

You must respond with exactly one of these formats:

ANSWER: {your complete answer}

or, if you cannot answer from the provided documents:

NOT_SURE: {specific reason why you cannot answer}

The NOT_SURE reason should be specific and actionable, explaining what information is missing or why the documents are insufficient.
```

#### Pipeline Generation Prompt (data exceeds context)

```
You need to answer a prompt over a large document set that exceeds the context limit.

PROMPT: {prompt}

DOCUMENTS: 
  Count: {count} items
  Total size: ~{tokens} tokens
  Sample (first document, truncated):
  ---
  {first 1000 chars of first doc}
  ---

{if log is non-empty}
PREVIOUS ATTEMPTS THAT FAILED:
{for each log entry}
  Attempt {n}:
    Pipeline: {description of flat_map_fn and reduce_fn}
    Result: {input_size} items → {output_size} items
    Failure reason: {failure_reason}
{end for}
{end if}

Write a pipeline to reduce this data so it can be processed. You must provide two Lua functions.

AVAILABLE BUILT-INS:
  chunk(item, size) → list[item]        -- split into chunks of ~size tokens
  fuzzy_match(item, pattern) → bool     -- fuzzy regex match
  exact_match(item, pattern) → bool     -- literal substring match
  semantic_score(item, query) → float   -- embedding similarity 0-1
  semantic_rank(items, query) → items   -- sort by similarity
  invoke(items, prompt) → Thought       -- recursive LLM call
  content(item) → string                -- get text content
  tokens(item) → int                    -- count tokens
  len(items) → int                      -- count items
  take(items, n) → items                -- first n items
  concat(a, b, ...) → items             -- combine lists

Respond with exactly this format:

FLAT_MAP_FN:
```lua
function(item)
  -- your code here
  -- return {} to drop, {item} to keep, or multiple items to expand
end
```

REDUCE_FN:
```lua
function(items)
  -- your code here  
  -- return the reduced list
end
```

REASONING: {one sentence explaining your strategy}
```

### CLI Interface

```
mfr-rlm - Answer prompts over large document sets

USAGE:
  mfr-rlm --docs <directory> --prompt <string> [options]

REQUIRED ARGUMENTS:
  --docs <path>         Directory containing input documents
  --prompt <string>     Question or instruction to answer

OPTIONS:
  --context-limit <n>   Token budget for base case (default: 100000)
  --max-attempts <n>    Max pipeline attempts per recursion level (default: 10)
  --max-depth <n>       Max recursion depth (default: 5)
  --model <string>      LLM model to use (default: claude-sonnet-4-20250514)
  --verbose             Print execution trace to stderr
  --output <path>       Write result to file (default: stdout)
  --help                Show this help message

OUTPUT:
  On success, prints the answer to stdout.
  On failure (NOT_SURE), prints the reason to stdout.

EXIT CODES:
  0   Success (answered)
  1   Could not answer (NOT_SURE)
  2   Error (invalid input, crash, etc.)

EXAMPLES:
  # Simple question
  mfr-rlm --docs ./reports --prompt "What were Q3 revenues?"
  
  # With options
  mfr-rlm --docs ./logs --prompt "What caused the outage?" \
    --context-limit 50000 --max-depth 3 --verbose
  
  # Save output
  mfr-rlm --docs ./papers --prompt "Summarize findings" --output summary.txt
```

### Implementation Guide

#### Project Structure

```
mfr-rlm/
├── main.lua              -- CLI entry point
├── invoke.lua            -- Core recursive algorithm  
├── sandbox.lua           -- Lua sandbox for executing pipelines
├── builtins.lua          -- Built-in functions (chunk, fuzzy_match, etc.)
├── llm.lua               -- LLM client (API calls, prompt formatting)
├── tokenizer.lua         -- Token counting
├── loader.lua            -- Document loading from directory
└── types.lua             -- Data type definitions
```

#### Key Implementation Details

##### 1. Lua Sandbox

Create a restricted Lua environment that only exposes the built-in functions. Remove access to `os`, `io`, `require`, `loadfile`, `dofile`, and any other dangerous functions.

```lua
local function create_sandbox(builtins)
  local env = {
    -- Safe Lua builtins
    pairs = pairs,
    ipairs = ipairs,
    type = type,
    tostring = tostring,
    tonumber = tonumber,
    
    -- Our builtins
    chunk = builtins.chunk,
    fuzzy_match = builtins.fuzzy_match,
    exact_match = builtins.exact_match,
    semantic_score = builtins.semantic_score,
    semantic_rank = builtins.semantic_rank,
    invoke = builtins.invoke,
    content = builtins.content,
    tokens = builtins.tokens,
    len = builtins.len,
    take = builtins.take,
    concat = builtins.concat,
  }
  return env
end
```

##### 2. Token Counting

Use a tokenizer appropriate for your target LLM (e.g., tiktoken for OpenAI models, or the Anthropic tokenizer for Claude). The token count should include:

- All document content
- The prompt
- Overhead for formatting (document tags, etc.)
- Reserve space for the response (~4000 tokens)

##### 3. Document Loading

On startup, recursively read all files from the `--docs` directory. Supported formats should include at minimum `.txt` and `.md`. Optionally support `.pdf` (via text extraction).

Each file becomes a `Document`:

```lua
{
  type = "document",
  content = file_contents,
  source = {
    file = relative_path,
    start = 0,
    ["end"] = #file_contents
  }
}
```

##### 4. Chunking

When `chunk(item, size)` splits a document:

- Split on sentence or paragraph boundaries when possible
- Each child chunk gets the same `source.file` but updated `start` and `end` offsets
- Aim for approximately `size` tokens per chunk, but don't split mid-word

##### 5. Semantic Operations

For `semantic_score` and `semantic_rank`, use a fast embedding model. Options:

- Local: `sentence-transformers` with a small model (e.g., `all-MiniLM-L6-v2`)
- Local sparse: SPLADE or BM25
- API: OpenAI embeddings, Cohere, etc.

Cache embeddings for items to avoid recomputation.

##### 6. Fuzzy Matching

For `fuzzy_match`, implement a forgiving regex that:

- Is case-insensitive by default
- Treats whitespace flexibly (multiple spaces, newlines treated as single separator)
- Optionally allows small edit distance for typo tolerance

A simple implementation could use Lua patterns with preprocessing, or integrate a proper regex library.

##### 7. Recursive Invocation

The `invoke(items, prompt)` builtin calls back into the main `invoke` function with:

- A fresh empty log
- The same context limit and max settings
- Incremented depth counter (to enforce max depth)

The result is wrapped in a `Thought`:

```lua
{
  type = "thought",
  value = result.text  -- or structured data if LLM returns it
}
```

##### 8. Flat Map Execution

```lua
local function flat_map(items, fn)
  local result = {}
  for _, item in ipairs(items) do
    local outputs = fn(item)
    for _, output in ipairs(outputs) do
      table.insert(result, output)
    end
  end
  return result
end
```

##### 9. Error Handling

If the LLM-generated Lua code fails to parse or execute:

1. Catch the error
2. Add it to the log as a failed attempt with reason "Code error: {message}"
3. Continue to next attempt

If the pipeline produces zero items and the recursion returns `NotSure`, that's handled normally by the retry loop.

##### 10. Verbose Output

In verbose mode, print to stderr:

```
[depth=0] invoke: 2000 docs, 8.2M tokens
[depth=0] attempt 1: flat_map(fuzzy filter "Q3") → reduce(identity)
[depth=0] result: 2000 → 89 docs
[depth=1] invoke: 89 docs, 620K tokens
[depth=1] attempt 1: flat_map(chunk 4000) → reduce(take 30)
[depth=1] result: 89 → 30 docs
[depth=2] invoke: 30 docs, 92K tokens
[depth=2] base case: calling LLM directly
[depth=2] result: ANSWER
[depth=1] result: ANSWER
[depth=0] result: ANSWER
```

### Example Execution Trace

**Input:**
- Docs: 500 financial reports, ~4M tokens
- Prompt: "What was the year-over-year revenue growth in Q3?"

**Execution:**

```
invoke(docs, prompt, log=[])
│
├─ tokens(docs) = 4M > CONTEXT_LIMIT
│
├─ Attempt 1:
│   LLM generates:
│     flat_map_fn: filter by fuzzy_match(item, "Q3.*revenue|revenue.*Q3")
│     reduce_fn: identity
│   
│   Execute: 500 docs → 67 docs (~480K tokens)
│   
│   └─ invoke(new_docs, prompt, log=[])
│      │
│      ├─ tokens = 480K > CONTEXT_LIMIT
│      │
│      ├─ Attempt 1:
│      │   flat_map_fn: chunk(item, 3000) then filter semantic_score > 0.4
│      │   reduce_fn: take(items, 40)
│      │   
│      │   Execute: 67 docs → 156 chunks → 40 chunks (~95K tokens)
│      │   
│      │   └─ invoke(new_docs, prompt, log=[])
│      │      │
│      │      ├─ tokens = 95K < CONTEXT_LIMIT
│      │      │
│      │      └─ Base case: llm_answer(docs, prompt)
│      │         └─ Returns: Answer("Q3 YoY revenue growth was 12.4%...")
│      │
│      └─ Returns: Answer(...)
│
└─ Returns: Answer("Q3 YoY revenue growth was 12.4%...")
```

### Testing

Test cases to implement:

1. **Base case**: Small document set that fits in context → direct answer
2. **Single recursion**: Medium set requiring one pipeline → answer
3. **Deep recursion**: Large set requiring multiple levels → answer
4. **Filter to empty**: Pipeline filters everything → NOT_SURE with reason
5. **Retry on failure**: First pipeline fails, second succeeds → answer
6. **Max attempts**: All attempts fail → NOT_SURE with last reason
7. **Max depth**: Recursion limit reached → appropriate error
8. **Code error**: LLM generates invalid Lua → retry with error in log
9. **Chunking**: Large documents get split correctly with source tracking
10. **Invoke builtin**: Pipeline uses `invoke()` for summarization → works

### Dependencies

Suggested dependencies (adjust based on implementation language):

- **Lua runtime**: LuaJIT or standard Lua 5.4
- **HTTP client**: For LLM API calls
- **JSON library**: For parsing LLM responses
- **Tokenizer**: tiktoken (if using OpenAI) or appropriate alternative
- **Embeddings**: sentence-transformers or API client
- **File I/O**: Standard library
- **CLI parsing**: argparse or similar
