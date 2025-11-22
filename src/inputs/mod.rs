use lopdf::Document;
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub enum InputError {
    FileNotFound(String),
    ReadError(String),
    PdfError(String),
    UnsupportedFormat(String),
}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputError::FileNotFound(path) => write!(f, "File not found: {path}"),
            InputError::ReadError(msg) => write!(f, "Error reading file: {msg}"),
            InputError::PdfError(msg) => write!(f, "Error processing PDF: {msg}"),
            InputError::UnsupportedFormat(msg) => write!(f, "Unsupported format: {msg}"),
        }
    }
}

impl std::error::Error for InputError {}

#[derive(Debug)]
pub struct Input {
    content: String,
}

impl Input {
    /// Load content from a file. Supports text files and PDFs.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, InputError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(InputError::FileNotFound(path.display().to_string()));
        }

        // Check if it's a PDF by extension
        if let Some(ext) = path.extension() {
            if ext.eq_ignore_ascii_case("pdf") {
                return Self::load_pdf(path);
            }
        }

        // Otherwise try to read as text
        Self::load_text(path)
    }

    /// Load a text file
    fn load_text<P: AsRef<Path>>(path: P) -> Result<Self, InputError> {
        let content =
            fs::read_to_string(path.as_ref()).map_err(|e| InputError::ReadError(e.to_string()))?;

        Ok(Input { content })
    }

    /// Load a PDF file and extract text
    fn load_pdf<P: AsRef<Path>>(path: P) -> Result<Self, InputError> {
        let doc = Document::load(path.as_ref())
            .map_err(|e| InputError::PdfError(format!("Failed to load PDF: {e}")))?;

        let mut content = String::new();

        // Extract text from all pages
        for page_num in 1..=doc.get_pages().len() {
            if let Ok(page_content) = doc.extract_text(&[page_num as u32]) {
                content.push_str(&page_content);
                content.push('\n');
            }
        }

        if content.is_empty() {
            return Err(InputError::PdfError(
                "No text could be extracted from PDF".to_string(),
            ));
        }

        Ok(Input { content })
    }

    /// Get the content as a string
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Create an Input from a string directly (for backwards compatibility or testing)
    pub fn from_string(content: String) -> Self {
        Input { content }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_text_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Hello, world!").unwrap();
        writeln!(file, "This is a test.").unwrap();

        let input = Input::from_file(file.path()).unwrap();
        assert!(input.content().contains("Hello, world!"));
        assert!(input.content().contains("This is a test."));
    }

    #[test]
    fn test_file_not_found() {
        let result = Input::from_file("/nonexistent/file.txt");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), InputError::FileNotFound(_)));
    }

    #[test]
    fn test_from_string() {
        let input = Input::from_string("Direct content".to_string());
        assert_eq!(input.content(), "Direct content");
    }
}
