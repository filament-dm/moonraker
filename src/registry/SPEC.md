# Registry

## Overview

The function registry is a directory of Lua functions that should be made available to the execution environment. A few functions are built-in, such as `print` (for capturing stdout), `llm_query` (for recursive LLM calls), and `token_*` (for context management). In addition to these built-ins, which are inserted directly into the execution environment, are present in all executions, and call Rust functionality, the registry provides Lua-only functions that can be added selectively depending on the user's use case.

The registry is generally loaded up at runtime for a RLM execution.

## Actions

`from_paths`: Initializes the registry with list of paths to search for functions. Iterates through these paths in order, loading functions into the registry's internal state. If two functions have the same name, the one appearing _later_ in the load order is the one that takes precedence. Thus, a user might set up a series of directories in priority order. Any attempt to override a built-in results in an error.

`from_map`: Similar to `from_paths` however rather than supplying a list of directories, receives a map of name to definition directly. This used primarily for testing.

## Queries

`iter_functions`: Outputs a map of function name to function definition for the Environment to use.

`system_prompt`: Outputs an addendum to the a system prompt that assembles the registry's functions. It declares what functions are available and their usages.

## Function Definition Conventions

We expect three things from function definitions:

1. The function name in the Lua code is the function name exposed to the execution environment. There may be multiple functions within each loaded `.lua` file.

2. The function _must_ be supplied with a triple dash comment before its definition. The entire function comment will then be supplied to the RLM.

3. Functions declared `local` are _not_ made available to the execution environment.

Any violation of these conventions results in a load error. `from_paths` and `from_map`, if they execute successfully, will guarantee that the Lua code is syntactically correct, has an accessible name, and accompanying function description and usage.
