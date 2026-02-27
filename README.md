🏥 HealthCare RAG – AI-Powered Medical Information Assistant

HealthCare RAG is an AI-powered, Retrieval-Augmented Generation (RAG) system designed to provide accurate, evidence-based healthcare information from a curated medical knowledge base. Instead of relying on generic model memory, the system retrieves relevant medical documents using vector embeddings and semantic search, then generates focused, informative responses.

🚀 Key Features
🔍 Semantic Search with FAISS for fast and accurate document retrieval
🧠 Sentence-Transformer embeddings for understanding medical queries
📚 Retrieval-Augmented Generation (RAG) architecture
⚡ FastAPI backend for real-time question answering
🖥️ Clean and minimal web-based UI
🔒 Offline-friendly knowledge base (no external medical scraping at runtime)

🏗️ Architecture
Medical documents are split into chunks
Chunks are converted into vector embeddings
FAISS indexes the vectors for similarity search
User queries retrieve the most relevant chunks
Retrieved context is synthesized into a clean, non-duplicated answer

🎯 Why RAG?
Traditional LLMs can hallucinate or provide outdated medical information.
This project solves that by grounding responses in verified healthcare documents, improving reliability and transparency.

🛠️ Tech Stack
Python
FastAPI
FAISS
Sentence Transformers
NumPy
HTML / CSS / JavaScript

⚠️ Disclaimer
This project is for educational purposes only and does not replace professional medical advice.
