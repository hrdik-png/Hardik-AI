# Hardik AI

Hardik AI is a personal RAG-based chatbot that retrieves information from a custom knowledge base and uses an LLM to generate context-grounded responses.

## Features

- Semantic search using BGE embeddings
- ChromaDB vector database
- Hybrid retrieval using semantic similarity and keyword matching
- Conversation-aware query rewriting
- Conversation history
- Context-grounded responses
- Streaming AI responses
- Flask web interface
- Unknown-information handling

## Tech Stack

- Python
- Flask
- ChromaDB
- Sentence Transformers
- BAAI/bge-small-en-v1.5
- Groq API
- OpenAI GPT-OSS 120B
- HTML
- CSS
- JavaScript

## How It Works

1. The user asks a question.
2. The question is rewritten using conversation history when necessary.
3. The rewritten query is converted into an embedding.
4. ChromaDB retrieves relevant documents.
5. Retrieved documents are ranked using semantic similarity and keyword matching.
6. The most relevant documents are provided as context to the language model.
7. The model generates a grounded response.
8. The response is streamed back to the web interface.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt