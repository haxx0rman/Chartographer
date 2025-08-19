# Chartographer Mindmap Generator

Chartographer is an advanced mindmap generation tool that leverages machine learning models to extract key concepts, topics, and details from text documents. It generates mindmaps in Mermaid syntax, HTML format, and Markdown outlines.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)

## Introduction

Chartographer is built to process various types of documents, including technical, scientific, business, legal, academic, and narrative texts. It offers enhanced features like semantic deduplication, reality checking, and multi-level processing.

### Key Components

1. **Mindmap Generator**: Core component for generating mindmaps from text or files.
2. **Token Usage Tracker**: Tracks token usage across various API calls to manage costs effectively.
3. **Document Type Detection**: Identifies the type of document to optimize mindmap generation based on content (technical, scientific, business, etc.).
4. **Chunking System**: Breaks down large documents into smaller chunks for processing.

## Features

- **Multi-Document Support**: Supports various types of documents including PDFs, text files, and code.
- **Advanced LLM Integration**: Utilizes OpenAI models to extract topics, subtopics, and details from text content.
- **Token Usage Tracking**: Keeps track of token usage for cost management.
- **Chunking Mechanism**: Handles large documents by splitting them into manageable chunks.

## Requirements

- Python 3.7+
- `openai` package
- `fuzzywuzzy` package

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/username/chartographer.git
    cd chartographer
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables by creating a `.env` file and adding your API keys and other configurations.

## Configuration

Chartographer uses a `ChartographerConfig` class to manage configuration settings. Here's an example of how you can configure it:

```python
from chartographer.mindmap_generator import ChartographerConfig

config = ChartographerConfig(
    api_provider="OLLAMA",  # or another provider like "OPENAI"
    llm_model="qwen3-coder:30b",
    max_tokens=50048,
    temperature=0.2,
    llm_host="http://100.95.157.120:11434"  # Example host
)
```

### Configuration Parameters

- **API Keys**: `openai_api_key`, `anthropic_api_key`, etc.
- **LLM Model and Hosting Information**: `llm_model`, `embedding_model`, `llm_host`
- **Directory Settings**: Paths for working directories, document storage, processed documents, etc.

## Usage

### Generating a Mindmap from File

1. Create a new Python script or Jupyter Notebook.
2. Import the necessary modules and initialize the configuration:

```python
import asyncio
from chartographer.mindmap_generator import MindmapGenerator, ChartographerConfig

async def main():
    # Set up logging configuration
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("chartographer.main")

    # Define the path to the input file and output directory
    input_file_path = "./workspace/input/SIE.md"
    output_directory = "./workspace/output/"

    # Configure BookWorm (adjust these settings as needed)
    config = ChartographerConfig(
        api_provider="OLLAMA",  # or another provider you're using
        llm_model="qwen3-coder:30b",
        max_tokens=50048,
        temperature=0.2,
        llm_host="http://100.95.157.120:11434"  # Example host, change as needed
    )

    # Initialize the MindmapGenerator with the configuration
    mindmap_generator = MindmapGenerator(config)

    try:
        # Generate and save the mind map from a file
        result = await mindmap_generator.generate_mindmap_from_file(input_file_path, output_directory)
        logger.info(f"Mindmap generated successfully: {result}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
```

### Generating a Mindmap from Text

If you have text content already, you can generate a mindmap directly from it:

```python
async def generate_from_text():
    config = ChartographerConfig(api_provider="OLLAMA", llm_model="qwen3-coder:30b")
    generator = MindmapGenerator(config)

    # Read the document content (or use raw text)
    with open("path/to/your/document.txt") as file:
        document_content = file.read()

    result = await generator.generate_mindmap_from_text(document_content, "document_filename")

# Run the function
asyncio.run(generate_from_text())
```

### Chunking Large Documents

If your document is large and needs chunking:

```python
config = ChartographerConfig(api_provider="OLLAMA", llm_model="qwen3-coder:30b")
generator = MindmapGenerator(config)

# Read the content of a large document
with open("path/to/large/document.txt") as file:
    text_content = file.read()

# Check if chunking is needed and create chunks
if generator._needs_chunking(text_content):
    chunks = generator._create_chunks(text_content)
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}: {chunk['text'][:50]}...")

# Generate mindmap from all chunks
result = await generator.generate_mindmap_from_text(" ".join([chunk["text"] for chunk in chunks]), "large_document")
```

## API Reference

### MindMap Generation

```python
class MindmapGenerator:
    def __init__(self, config: ChartographerConfig):
        # Initialize with configuration settings

    async def generate_mindmap_from_file(self, filename: str, output_dir: str) -> Mindmap:
        """Generate a mindmap from a file and save outputs to the specified directory."""

    async def generate_mindmap_from_text(self, text_content: str, filename) -> Mindmap:
        """Generate a comprehensive mindmap with advanced features and chunking support."""
```

### Token Usage Tracking

```python
class TokenUsageTracker:
    def __init__(self):
        # Initialize token usage tracker

    def add_usage(self, category: str, input_tokens: int, output_tokens: int):
        """Add token usage for a specific category."""

    def get_total_tokens(self) -> Dict[str, int]:
        """Get total token usage across all categories."""

    def print_usage_report(self):
        """Print detailed usage report."""
```

### Document Type Detection

```python
async def _detect_document_type(self, text_content: str) -> DocumentType:
    """Detect the type of document based on content keywords and patterns."""
```

## Error Handling

Chartographer raises custom exceptions when errors occur during mindmap generation:

- `MindMapGenerationError`: Raised for general mindmap generation failures.
- `FileNotFoundError`: Raised if a specified file is not found.

## Logging

All actions within the Chartographer library are logged using Python's built-in logging module. Logs include:

- Information messages about document processing and mindmap generation
- Error messages with detailed stack traces when exceptions occur
- Debug information for troubleshooting (if enabled)

### Configuring Logging

To enable detailed debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any issues or suggestions, please open an issue on GitHub.