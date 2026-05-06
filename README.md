# LLM-based-PDF-Chatbot-using-RAG
## Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDFs and ask questions from documents using semantic search and LLM-powered responses.

## Features
- PDF upload support
- Semantic search using embeddings
- Vector database retrieval
- Interactive question answering
- Streamlit-based UI

## Tech Stack
- Python
- LangChain
- FAISS
- Hugging Face Embeddings
- OpenAI API
- Streamlit

## How It Works
1. PDF is uploaded
2. Text is extracted and chunked
3. Embeddings are generated
4. FAISS vector store indexes chunks
5. Relevant chunks retrieved using semantic similarity
6. LLM generates final response

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
