# Agent Instructions

This repository contains the Moonraker project, a Lua REPL agent system.

## Setup Requirements

For integration tests and full functionality, you must ensure the following are available in the environment:

1.  **Ollama**: The `ollama` binary must be installed and running.
2.  **Model**: The `qwen3:30b` model must be pulled.

### Installation Instructions (for Agents)

If you are setting up the environment, perform the following steps:

1.  **Install Ollama**:
    Download and install Ollama from [ollama.com](https://ollama.com).
    On Linux/Mac (typical):
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

2.  **Start Ollama**:
    Ensure the Ollama server is running in the background.
    ```bash
    ollama serve &
    ```

3.  **Pull the Model**:
    Download the required model for tests.
    ```bash
    ollama pull qwen3:30b
    ```

## Running Tests

To run the standard test suite (excluding integration tests that require Ollama):
```bash
cargo test
```

To run the full test suite (including integration tests):
```bash
cargo test --features integration
```
