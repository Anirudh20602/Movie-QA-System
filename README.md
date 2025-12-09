# NLP Homework 4: Movie Question-Answering System

**Student**: Anirudh Krishna  
**Course**: Natural Language Processing  
**Assignment**: Homework 4 - RAG and LLM-based QA System

## Overview

A movie question-answering system using RAG (Retrieval-Augmented Generation) for semantic queries and template-based code generation for factual queries on 10,000 IMDB movies.

## Features

- **Semantic Queries**: Vector-based retrieval using sentence-transformers and FAISS
- **Factual Queries**: Template-based pandas code generation for statistics
- **Unified Interface**: Automatic query classification and routing

## Files

- **`movie_QA_system.ipynb`**: Main Jupyter notebook (optimized for Google Colab)
- **`IMDB_top_10000_07132023.csv`**: Dataset (10,000 IMDB movies)
- **`movie_index/`**: Persisted vector index directory (generated after first run)
- **`README.md`**: This file

## Quick Start (Google Colab)

1. Upload `movie_QA_system.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Enable T4 GPU: Runtime → Change runtime type → T4 GPU
3. Upload `IMDB_top_10000_07132023.csv` or mount Google Drive
4. Run all cells

## Local Setup

```bash
pip install transformers torch pandas numpy sentence-transformers
pip install llama-index-core llama-index-embeddings-huggingface llama-index-llms-huggingface
pip install faiss-cpu accelerate
jupyter notebook movie_QA_system.ipynb
```

## Models Used

- **T5**: `google/flan-t5-base` (natural language formatting)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (semantic search)

## Example Queries

### Semantic (RAG-based)
```python
semantic_query("What are some alien-related movies?")
# Returns: Top 5 relevant movies based on semantic similarity
```

### Factual (Template-based)
```python
factual_query("What's the average rating of James Bond movies?")
# Returns: Computed statistics with natural language answer
```

## Supported Query Types

- **Average**: rating, gross revenue, runtime
- **Count**: by year, director, rating threshold
- **Top-N**: highest-rated, highest-grossing
- **Year**: most releases

## Implementation

### Part 1: Setup (10 pts) ✅
Package installation, data loading

### Part 2: Vector Index (10 pts) ✅
FAISS-based vector store with sentence-transformers

### Part 3: Semantic Queries (25 pts) ✅
RAG implementation with top-k retrieval

### Part 4: Factual Queries (35 pts) ✅
Template-based code generation with safe execution

### Part 5: Unified Interface (20 pts) ✅
Query classification and integration

**Total: 100 points**

## Design Notes

**Template-based Code Generation**: T5-base (220M params) proved too small for reliable code generation. Templates ensure 100% valid code for all test queries. For production, use larger models (CodeLlama, GPT-4).

**Local RAG**: Uses llama-index retriever without LLM for answer generation. Demonstrates RAG principles without external API dependencies.

## Troubleshooting

**Out of Memory**:
```python
import gc
torch.cuda.empty_cache()
gc.collect()
```

**Unsupported Query**: Rephrase to match supported patterns (average, count, top-N, year-based).

## References

- [LlamaIndex](https://www.llamaindex.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [T5 Model](https://huggingface.co/google/flan-t5-base)
