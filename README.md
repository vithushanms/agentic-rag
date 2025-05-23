# RAG Proof of Concept

A demonstration of a Retrieval-Augmented Generation (RAG) approach for document querying and analysis.

## Demo Video

[![RAG Demo](https://img.youtube.com/vi/8Stx24uSp3U/0.jpg)](https://youtu.be/8Stx24uSp3U)

## Overview

This project implements a complete RAG pipeline with two main components:

1. **Document Processing & Vectorization** (`pdf_processor.py`) - Extracts text from PDF documents and creates vector embeddings for semantic search
2. **Agent-based Query Interface** (`agent.ipynb`) - An intelligent agent that can query documents using a retrieval tool and provide contextual answers

## Architecture

### PDF Processing Pipeline
- **Input**: PDF documents in the `raw_files/` directory
- **Processing**: Text extraction using PyPDF2, chunking with LangChain's RecursiveCharacterTextSplitter
- **Vectorization**: OpenAI's text-embedding-ada-002 model with FAISS vector store
- **Output**: Searchable vector database stored in `vector_store/`

### Agent Interface
- **Retrieval Tool**: Custom LangChain tool that queries the vector store
- **Agent**: LangGraph-based ReAct agent with conversational memory
- **LLM**: GPT-4.1 for reasoning and response generation
- **Context**: General document analysis and querying

## Setup

### Prerequisites
- Miniconda or Anaconda
- OpenAI API key

### Setup

1. **Create and activate conda environment:**
```bash
conda env create -f environment.yml
conda activate rag_poc
```

2. **Install Poetry in the environment:**
```bash
pip install poetry
poetry install
```

3. **Set up environment variables:**
```bash
cp .env.template .env
# Edit .env with your OpenAI API key
```

4. **Run the processing pipeline:**
```bash
python pdf_processor.py
```

This will process PDF files in `raw_files/` and create the vector store.

5. **View results:**
Open and run `agent.ipynb` to interact with the processed documents and see the RAG system in action.

## Example Queries

The system can handle various document-related queries:

- "Summarize the key findings from the document"
- "What are the main topics discussed?"
- "Extract specific data points from the analysis"
- "Provide insights based on the document content"

## Project Structure

```
rag_poc/
├── pdf_processor.py      # Document processing and vectorization
├── agent.ipynb          # Interactive agent interface
├── raw_files/           # Input PDF documents
├── vector_store/        # FAISS vector database
├── pyproject.toml       # Poetry dependencies
├── environment.yml      # Conda environment (if needed)
└── .env.template        # Environment variable template
```

## Key Dependencies

- **LangChain**: Framework for LLM applications and document processing
- **LangGraph**: Agent framework with tool integration
- **FAISS**: Vector similarity search
- **OpenAI**: LLM and embedding models
- **PyPDF2**: PDF text extraction

## Use Cases

This RAG system can be applied to various document analysis scenarios:

- Technical document analysis
- Research paper querying
- Report summarization
- Data extraction from structured documents
- Interactive document exploration

## Configuration

Key environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `RAG_VECTORSTORE_PATH`: Path to vector store (default: "vector_store")

## Contributing

This is a proof-of-concept demonstration. For production use, consider:
- Adding error handling and logging
- Implementing document update mechanisms
- Adding authentication and access controls
- Optimizing embedding and retrieval parameters
- Adding support for other document formats