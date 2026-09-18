# NTC Policy RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for National Telecommunication Corporation (NTC) policies, rules, and regulations.

## Features

- **PDF Ingestion**: Automatically extracts text from PDF documents using PyMuPDF4LLM
- **OCR Support**: Handles scanned PDFs using Docling OCR
- **Vector Search**: Uses Pinecone for fast similarity search
- **Local Embeddings**: Uses Ollama with nomic-embed-text model
- **Groq LLM**: Fast inference using Groq's API

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 3. Start Ollama

```bash
ollama serve
ollama pull nomic-embed-text
```

### 4. Create Pinecone index

Create an index named `rag-index` with dimension 768.

## Usage

### Ingest PDFs

```bash
# Ingest all PDFs from data/documents/
python src/ingest.py

# Ingest a specific PDF
python src/ingest.py --file data/documents/your_file.pdf
```

### Run the chatbot

```bash
python src/main.py
```

### Delete old vectors

```bash
# Delete vectors for a specific document
python src/delete_old_vector.py --name DocumentName

# Delete all vectors
python src/delete_old_vector.py --all
```

## Project Structure

```
.
├── data/
│   └── documents/          # PDF source files
├── src/
│   ├── config.py           # Configuration and API keys
│   ├── chunking.py         # Text splitting logic
│   ├── ingest.py           # PDF extraction and ingestion
│   ├── rag.py              # Query and answer generation
│   ├── main.py             # Interactive chat interface
│   ├── delete_old_vector.py # Vector deletion utility
│   ├── extract_preview.py  # Preview extracted chunks
│   └── test_pipeline.py    # Pipeline validation tests
├── requirements.txt
├── .env                    # Environment variables (not tracked)
└── README.md
```

## How It Works

1. **Ingestion**: PDFs are extracted to markdown, chunked, embedded using Ollama, and stored in Pinecone
2. **Query**: User question is embedded and searched against Pinecone
3. **Generation**: Relevant chunks are sent to Groq LLM with the question to generate an answer

## License

MIT
