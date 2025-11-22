use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::{ollama, openrouter};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::json;
use std::error::Error;

/// Trait for inputs to language models
pub trait LmInput {
    /// Format the input as a string prompt
    fn format(&self) -> String;
}

/// Trait for parsing text output into structured format.
///
/// Uses manual XML parsing instead of structured output (see README.md "Testing" section).
pub trait OutputParser: Sized {
    /// Parse the text output into the structured type
    fn parse(text: &str) -> Result<Self, Box<dyn Error>>;
}

/// Trait for language model providers that can generate structured outputs
#[async_trait]
pub trait LmProvider<I: LmInput + Send + 'static, O: DeserializeOwned + JsonSchema + Send + 'static>
{
    /// Set the system prompt for the provider
    fn with_system(self, prompt: String) -> Self;

    /// Generate a structured output from the given input
    async fn generate(&self, input: I) -> Result<O, Box<dyn Error>>;
}

/// Provider type enum
pub enum ProviderType {
    Ollama(ollama::Client),
    Openrouter(openrouter::Client),
}

/// Rig provider implementation (supports Ollama and OpenRouter)
pub struct RigProvider {
    client: ProviderType,
    model: String,
    system_prompt: Option<String>,
    /// API key for OpenRouter (if applicable)
    api_key: Option<String>,
}

impl RigProvider {
    /// Create a new Rig provider with Ollama backend and custom system prompt
    pub fn new_ollama_with_system(model: String, system_prompt: String) -> Self {
        Self {
            client: ProviderType::Ollama(ollama::Client::new()),
            model,
            system_prompt: Some(system_prompt),
            api_key: None,
        }
    }

    /// Create a new Rig provider with OpenRouter backend, custom system prompt, and provided API key
    pub fn new_openrouter_with_system_and_key(
        model: String,
        system_prompt: String,
        api_key: String,
    ) -> Self {
        Self {
            client: ProviderType::Openrouter(openrouter::Client::new(&api_key)),
            model,
            system_prompt: Some(system_prompt),
            api_key: Some(api_key),
        }
    }

    /// Create an LlmClient for the REPL environment from this provider
    pub fn to_llm_client(&self) -> Result<crate::environment::LlmClient, Box<dyn Error>> {
        match &self.client {
            ProviderType::Ollama(_) => {
                Ok(crate::environment::LlmClient::Ollama(self.model.clone()))
            }
            ProviderType::Openrouter(_) => {
                let api_key = self.api_key.clone().ok_or("OpenRouter API key not set")?;
                Ok(crate::environment::LlmClient::Openrouter(
                    self.model.clone(),
                    api_key,
                ))
            }
        }
    }
}

#[async_trait]
impl<I, O> LmProvider<I, O> for RigProvider
where
    I: LmInput + Send + 'static,
    O: DeserializeOwned + JsonSchema + OutputParser + Send + 'static,
{
    fn with_system(self, _prompt: String) -> Self {
        // Extract the model name from the existing agent
        // Since we can't easily modify an existing agent, we'll create a new one
        // This is a workaround - in practice, we should construct the agent with the system prompt upfront
        // For now, we'll just return self and rely on new_ollama_with_system being used instead
        self
    }

    async fn generate(&self, input: I) -> Result<O, Box<dyn Error>> {
        // Get the formatted prompt from the input
        let user_prompt = input.format();

        // Build the agent based on the provider type
        let response: String = match &self.client {
            ProviderType::Ollama(client) => {
                let agent = if let Some(system_prompt) = &self.system_prompt {
                    client
                        .agent(&self.model)
                        .preamble(system_prompt)
                        .additional_params(json!({"think": false}))
                        .build()
                } else {
                    client
                        .agent(&self.model)
                        .additional_params(json!({"think": false}))
                        .build()
                };
                agent.prompt(&user_prompt).await?
            }
            ProviderType::Openrouter(client) => {
                let agent = if let Some(system_prompt) = &self.system_prompt {
                    client.agent(&self.model).preamble(system_prompt).build()
                } else {
                    client.agent(&self.model).build()
                };
                agent.prompt(&user_prompt).await?
            }
        };

        // Parse the text response using the OutputParser trait
        let parsed: O = O::parse(&response)?;

        Ok(parsed)
    }
}

/// Recursive Language Model implementation
pub struct Rlm<P>
where
    P: LmProvider<crate::repl::Repl, crate::repl::Cell>,
{
    provider: P,
    repl: crate::repl::Repl,
}

impl<P> Rlm<P>
where
    P: LmProvider<crate::repl::Repl, crate::repl::Cell>,
{
    /// Create a new Rlm with the given provider and initial prompt/context
    pub fn new(
        provider: P,
        prompt: String,
        context: String,
        model: String,
        client: crate::environment::LlmClient,
    ) -> Result<Self, Box<dyn Error>> {
        let repl = crate::repl::Repl::new(prompt, context.as_str(), model, client)
            .map_err(|e| format!("Failed to create REPL: {e}"))?;

        Ok(Self { provider, repl })
    }

    /// Perform a single step: generate a Cell from the LM, execute it, and return the executed Cell
    pub async fn step(&mut self) -> Result<crate::repl::Cell, Box<dyn Error>> {
        // Create a snapshot of the REPL for input
        let repl_snapshot = self
            .repl
            .snapshot()
            .map_err(|e| format!("Failed to create REPL snapshot: {e}"))?;

        // Generate a partial Cell (with output set to None) from the LM
        let cell = self.provider.generate(repl_snapshot).await?;

        // Preserve the final flag from the LM-generated cell
        let is_final = cell.r#final;

        // Execute the code in the REPL
        self.repl.eval(&cell.comment, &cell.code);

        // Return the executed cell (with output computed) and restore the final flag
        let mut executed_cell = self.repl.entries.last().unwrap().clone();
        executed_cell.r#final = is_final;
        Ok(executed_cell)
    }

    /// Create an iterator that yields executed Cells for up to max_iterations steps
    pub fn execute(&mut self, max_iterations: usize) -> RlmIterator<P> {
        RlmIterator {
            rlm: self,
            remaining: max_iterations,
        }
    }

    /// Return the output of the final Cell, if it exists
    pub fn final_output(&self) -> Option<String> {
        self.repl
            .entries
            .last()
            .and_then(|cell| cell.output.clone())
    }
}

/// Iterator for executing RLM steps
pub struct RlmIterator<'a, P>
where
    P: LmProvider<crate::repl::Repl, crate::repl::Cell>,
{
    rlm: &'a mut Rlm<P>,
    remaining: usize,
}

impl<'a, P> RlmIterator<'a, P>
where
    P: LmProvider<crate::repl::Repl, crate::repl::Cell>,
{
    /// Get the next Cell by executing one step
    pub async fn next(&mut self) -> Option<Result<crate::repl::Cell, Box<dyn Error>>> {
        if self.remaining == 0 {
            return None;
        }

        self.remaining -= 1;
        Some(self.rlm.step().await)
    }

    /// Get the number of remaining iterations
    pub fn remaining(&self) -> usize {
        self.remaining
    }
}
