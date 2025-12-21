# Task 3 — Vector Index Creation

## Overview
This step creates a vector index for the custom knowledge base using local embeddings and FAISS.

## Embedding Model
- sentence-transformers/all-MiniLM-L6-v2
- Embedding size: 384
- Local execution (CPU)

## Knowledge Base
- Source: `knowledge_base/`
- 35 unique documents with renamed fictional entities
- Created in Task 2

## Chunking
- Splitter: RecursiveCharacterTextSplitter
- Chunk size: 1000
- Overlap: 150
- Metadata: source, title, chunk_id

## Vector Store
- FAISS (local)
- Index artifacts:
  - index/faiss.index
  - index/index.pkl

## Index Statistics
- Documents: 35
- Chunks: 1847
- Indexing time: ~20.55 seconds

## Build Index
```bash
python build_index.py
