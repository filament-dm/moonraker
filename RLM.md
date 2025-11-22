# Recursive Language Model (RLM) Algorithm

**Original work**: Zhang, A., & Khattab, O. (2025). [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/).

This document provides an implementation overview of their algorithm.

## Overview

RLM is an inference strategy that enables language models to handle unbounded input context by decomposing queries and recursively interacting with context through a REPL environment, rather than passing massive contexts directly to the model.

## Core Architecture

**Input**: A query and potentially huge context (e.g., millions of tokens)

**Output**: Answer to the query, produced through iterative REPL interactions

**Key Innovation**: The context is stored as a Python variable in a REPL environment. The root LM only sees the query and knows the context exists as a variable—it must write Python code to interact with it.

## Algorithm Flow

### 1. Initialization
- Create a Python REPL environment
- Store the full context in a variable called `context`
- Provide the root LM with a system prompt explaining:
  - Context is available as a variable (with its size)
  - It can write Python code to peek at, search, partition, or process the context
  - It can make recursive `rlm(sub_query, sub_context)` calls within the REPL
  - It signals completion with `FINAL(answer)` or `FINAL_VAR(variable_name)`

### 2. Execution Loop
The root LM (at depth 0) enters an iterative loop:

1. **LM generates response**: The model decides what to do next—it might peek at the context, grep for patterns, partition data, or make recursive calls
2. **Extract code blocks**: Parse any Python code from the LM's response
3. **Execute in REPL**: Run the code in the REPL environment
   - If code contains `rlm(query, context_var)`, spawn a recursive LM call at depth+1
   - Recursive calls get their own fresh REPL with the sub-context
4. **Return results**: Show execution output/errors back to the LM
5. **Repeat**: Continue until LM outputs `FINAL(answer)` or max iterations reached

### 3. Recursive Calls
When the LM writes `result = rlm("what type is this question?", chunk)`:
- Parse out the sub-query and sub-context variable
- Create a new RLM instance at depth+1 with its own REPL
- Execute the full RLM loop recursively for this sub-query
- Return the result back to the parent LM's REPL

### 4. Termination
The algorithm terminates when:
- The LM outputs `FINAL(answer)` with a direct answer, or
- The LM outputs `FINAL_VAR(var_name)` pointing to a REPL variable containing the answer
- Extract and return this final answer to the user

## Common Emergent Patterns

The LM learns to employ several strategies:

- **Peeking**: `context[:2000]` to understand structure before processing
- **Grepping**: `[line for line in context.split('\n') if 'user_id: 67144' in line]` to filter
- **Partition + Map**: Split context into chunks, make parallel recursive calls for semantic tasks (e.g., "label each question"), then aggregate results
- **Summarization**: Recursively condense portions of context

## Key Properties

- **Single API**: `rlm_completion(query, context)` is a drop-in replacement for `llm_completion(query, context)`
- **No context limits**: No single LM call sees the entire huge context
- **Interpretable**: Full REPL history shows exactly how the LM decomposed the problem
- **Learnable**: The decomposition strategy can be optimized via RL training

## Example System Prompt

The following is a system prompt from a prototype implementation:

You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {{header}} section: {{info}}")
    buffers.append(f"{{header}}: {{summary}}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
