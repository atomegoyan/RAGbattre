# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
streamlit run retrieval_app/app.py
```

### Data Pipeline Commands
```bash
# 1. Split corpus into chunks (required before embeddings)
python scripts/split_corpus_cs1.py

# 2. Generate embeddings (creates ChromaDB collections)
python scripts/embeddings/generate_embeddings_cs1.py

# 3. Generate additional chunks if needed
python scripts/generate_chunks.py
```

### Testing and Development
```bash
# Run development tests
python retrieval_app/test.py

# Launch Jupyter notebook for experimentation
jupyter notebook notebooks/rag.ipynb
```

## Architecture Overview

This is a **Parliamentary Debate Analysis RAG system** for French historical documents (1881). The architecture follows a three-stage pipeline:

### 1. Data Processing Pipeline
- **Raw documents** (`data/corpus/`) → **Chunking** (`scripts/split_corpus_cs1.py`) → **Embeddings** (`scripts/embeddings/generate_embeddings_cs1.py`) → **ChromaDB storage** (`data/embeddings_cs1/`)
- Uses custom chunking strategy for parliamentary text with speaker identification via regex
- Embedding model: `Alibaba-NLP/gte-multilingual-base` with L2 distance and HNSW indexing

### 2. Core Application (`retrieval_app/`)
- **app.py**: Streamlit interface with three modes (RAG, Document Retrieval, Chat)
- **retrieval/core.py**: Document search engine with multiple retrieval strategies
- **ollama_utils.py**: LLM integration for generation
- **config.py**: Centralized configuration management

### 3. Retrieval Strategies
- **Naive semantic search**: Standard vector similarity
- **Regex filtering**: Parliamentary speaker-based filtering
- **Document reranking**: Uses `BAAI/bge-reranker-v2-m3` for improved relevance

## Key Configuration

### Default Settings (config.py)
- **Collection**: `"1881-01-20"` (parliamentary session date)
- **Embedding Model**: `"Alibaba-NLP/gte-multilingual-base"`
- **Generation Model**: `"llama3.2:1b"` (configurable via Ollama)
- **Device**: CPU (can be changed to CUDA in embedding scripts)
- **Chunk Size**: 10,000 characters with regex-based speaker separation

### Data Structure Requirements
```
data/
├── corpus/                    # Raw parliamentary documents
├── corpus_splitted_cs1/       # Processed chunks by session
├── embeddings_cs1/            # ChromaDB persistent storage
└── questions_strat1.jsonl    # Example evaluation questions
```

## Important Implementation Details

### Parliamentary Text Processing
- Custom regex patterns in chunking for speaker identification (`M. [Name]` format)
- Hierarchical chunking preserves debate structure and speaker attribution
- Source attribution extracts exact text matches from original documents

### ChromaDB Collections
- Each parliamentary session becomes a separate collection (e.g., `"1881-01-20"`)
- Collections must be created via `generate_embeddings_cs1.py` before querying
- Uses HNSW indexing with L2 distance for efficient similarity search

### Ollama Integration
- Local LLM deployment required for generation
- Models are downloaded automatically via Ollama API
- Supports model switching through the Streamlit interface
- System prompts optimized for parliamentary document analysis

### Dependencies
Key packages: `streamlit`, `chromadb`, `sentence-transformers`, `ollama`, `transformers` (for reranking), `regex` (for parliamentary text parsing)

## Development Notes

- The application expects Ollama to be running locally for LLM functionality
- Embedding generation requires GPU access for optimal performance (modify device settings in scripts)
- New parliamentary sessions require running the full data pipeline (split → embed → store)
- The `/notebooks/rag.ipynb` is useful for testing retrieval strategies before implementing in the main app