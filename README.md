🤖 RAG AI Interviewer

An AI-powered Machine Learning & Deep Learning Interview Assistant built using Retrieval-Augmented Generation (RAG).

The application retrieves relevant information from a custom ML/DL knowledge base using ChromaDB and generates contextual answers using a local Qwen 2.5 1.5B Instruct model.

✨ Features

- 🔍 Retrieval-Augmented Generation (RAG)
- 📚 ML & DL knowledge base
- 🗃️ ChromaDB vector database
- 🧠 Qwen 2.5 1.5B Instruct
- 🔢 Sentence Transformers embeddings
- 💬 Interactive Streamlit UI
- 🔒 No paid API required

🛠️ Tech Stack

Python | Streamlit | ChromaDB | Sentence Transformers | Hugging Face Transformers | PyTorch | Qwen

🔄 RAG Pipeline

User Question
     ↓
Embedding
     ↓
ChromaDB Retrieval
     ↓
Relevant Context
     ↓
Qwen LLM
     ↓
Final Answer

▶️ Run Locally

pip install -r requirements.txt
streamlit run app.py

🎯 Purpose

Built as an AI/ML project to explore RAG, vector databases, embeddings, semantic search, and local LLMs for technical interview preparation.

👩‍💻 Author

Vedika Dwivedi
B.Tech – Electronics & Communication Engineering

⭐ If you find this project useful, consider giving it a star!
