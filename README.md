# Music Discovery

A semantic search system for music reviews using vector embeddings and SQLite.
The aim of this project is to help users discover new music based on other people's reviews, agnostic of styles or genres.
This project is using a dataset of PitchFork reviews, which can be found on [Kaggle](https://www.kaggle.com/datasets/nolanbconaway/pitchfork-data).

## Overview

This project enables semantic search over music reviews by:
- Embedding reviews into vector space using sentence transformers
- Storing embeddings in SQLite with the `sqlite_vec` extension
- Generating tags for reviews using Ollama models
- Performing k-nearest neighbor search to find similar reviews

## Features

- Vector embeddings of music reviews using `BAAI/bge-base-en-v1.5`
- Semantic search with k-NN queries
- AI-generated tags (genres, emotions, instruments) using Ollama
- GPU acceleration support (CUDA/ROCm)

## Usage

- `embedder.py` - Generate embeddings and populate the vector table
- `search.py` - Perform semantic search queries
- `preparedb.v2.py` - Generate tags for reviews using Ollama (async batch processing)

## Requirements

- Python 3.x
- SQLite with `sqlite_vec` extension
- Sentence Transformers
- Ollama (for tag generation)
- PyTorch (with CUDA/ROCm support optional)

## Todo

- [x] Embed reviews into vectors and insert into virtual table (DONE)
- [x] Implement a search function that takes an input and returns the k nearest neighbours (DONE)
- [x] Add tags for each review using ollama model to improve accuracy (DONE)
- [ ] Chunk reviews into smaller chunks to further improve accuracy

