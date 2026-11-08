# LangChain Length-Based Text Splitter

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-LangChain-green)
![Text Processing](https://img.shields.io/badge/Focus-Text%20Splitting-orange)
![GenAI](https://img.shields.io/badge/Domain-Generative%20AI-purple)

A practical implementation of **LangChain's `CharacterTextSplitter`** for dividing large documents into smaller, manageable text chunks.

This project explores how chunk size, separators, and chunk overlap influence document splitting and why effective chunking is important when building Retrieval-Augmented Generation (RAG) applications.

---

## Project Overview

Large language models have context-window limitations, which means processing an entire document at once is not always practical.

In RAG applications, large documents are commonly divided into smaller chunks before generating embeddings and storing them in a vector database.

LangChain provides text splitters to simplify this process.

This project focuses on understanding length-based text splitting and demonstrates how to:

- Divide large documents into smaller text chunks.
- Configure chunk size to control the maximum chunk length.
- Use separators to define preferred splitting boundaries.
- Apply chunk overlap to retain context between consecutive chunks.
- Transform LangChain `Document` objects into smaller documents.
- Prepare text chunks for embedding generation and vector storage.

## Key Concepts and Implementations

### 1. CharacterTextSplitter Integration

`CharacterTextSplitter` is a LangChain text splitter that divides text using a specified separator.

It provides control over how text is segmented into smaller pieces.

Example:

```python
from langchain_text_splitters import CharacterTextSplitter

text = """
Generative AI is transforming how applications
process and generate information.

LangChain provides reusable components
for building LLM-powered applications.
"""

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(chunk)
```

### 2. Understanding Chunk Size

The `chunk_size` parameter controls the target maximum size of each text chunk.

For example:

```python
chunk_size=500
```

This configures the splitter to target chunks of approximately 500 characters when using the default character-based length measurement.

Choosing an appropriate chunk size is important:

- Very small chunks may lose useful context.
- Very large chunks may contain unnecessary information.
- The best size depends on the document structure and retrieval requirements.

**Note:** `CharacterTextSplitter` uses character-based length measurement by default, not token counting.

### 3. Separator Configuration

Separators determine where the splitter looks for boundaries when dividing text.

Examples include:

| Separator | Purpose |
|---|---|
| `"\n\n"` | Paragraph-based splitting |
| `"\n"` | Line-based splitting |
| `" "` | Word-boundary splitting |
| `"."` | Sentence-oriented splitting |

Example:

```python
separator = "\n\n"
```

This tells the splitter to use paragraph boundaries when creating chunks.

The separator choice can influence how much contextual information remains together.

### 4. Understanding Chunk Overlap

Chunk overlap maintains a small amount of shared text between consecutive chunks.

For example, if one chunk ends with a particular sentence, some of that content may also appear at the beginning of the next chunk.

This helps reduce the loss of context when important information spans multiple chunks.

Example:

```python
chunk_overlap=50
```

This configures an overlap target of 50 characters.

The actual overlap depends on the available separator boundaries and the splitting configuration.

### 5. Document Transformation

LangChain text splitters can process both plain text and `Document` objects.

For example:

```python
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

document = Document(
    page_content="Your long document content goes here.",
    metadata={"source": "sample.txt"}
)

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents([document])

for chunk in chunks:
    print(chunk.page_content)
    print(chunk.metadata)
```

When splitting documents, metadata is generally carried over to the resulting chunks.

This is useful for maintaining source information throughout a RAG pipeline.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | LLM application framework |
| langchain-text-splitters | Text splitting utilities |
| langchain-core | Document objects and core components |
| python-dotenv | Environment variable management, if required |

---

## Project Structure

```text
length-based-text-splitter/
│
├── length_based.py
├── requirements.txt
└── README.md
```

Adjust the structure if your repository contains additional files or sample documents.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mani787060/length-based-text-splitter.git
cd length-based-text-splitter
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -U langchain-core langchain-text-splitters python-dotenv
```

If your implementation imports additional LangChain packages, install those as well.

### 4. Run the Implementation

```bash
python length_based.py
```

The script should demonstrate how the configured text splitter divides the input text or documents into smaller chunks.

---

## How Text Splitting Fits into RAG

Text splitting is an important stage between document loading and embedding generation.

A simplified RAG workflow:

```text
      Source Documents
             |
             v
       Document Loader
             |
             v
       Text Splitter
             |
             v
       Smaller Chunks
             |
             v
     Embedding Generation
             |
             v
       Vector Database
             |
             v
          Retrieval
             |
             v
        LLM Response
```

This repository focuses on the **text-splitting stage** of the pipeline.

The resulting chunks can be passed to an embedding model and stored in a vector database for semantic retrieval.

---

## Learning Outcomes

By exploring this project, you can understand:

- Why document chunking is necessary in RAG applications.
- How `CharacterTextSplitter` works.
- How chunk size affects document segmentation.
- How separators influence splitting boundaries.
- Why chunk overlap helps preserve context.
- How to split LangChain `Document` objects.
- How metadata is maintained during document transformation.

## Future Improvements

- Compare `CharacterTextSplitter` with `RecursiveCharacterTextSplitter`.
- Experiment with different chunk sizes and overlap values.
- Explore token-based splitting.
- Integrate embedding generation.
- Store chunks in FAISS or Chroma.
- Build a complete RAG-based question-answering application.
- Evaluate how chunking strategies influence retrieval quality.
