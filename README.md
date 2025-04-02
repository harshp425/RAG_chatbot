# RAG Chatbot

## Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot that provides intelligent responses by retrieving relevant information from a knowledge base and generating context-aware answers. It is designed to be run without using paid model APIs, making it accessible and cost-effective. The chatbot aims to deliver high-quality responses efficiently, even when running on a CPU.

## Technologies Used
- **Languages**: Python
- **Libraries**:
  - `Flask` (for serving the chatbot as a web application)
  - `LangChain` (for implementing RAG)
  - `FAISS` (for efficient vector search and retrieval)
  - `Transformers` (for working with open-source LLMs)
  - `Hugging Face` (for loading pre-trained models)
  - `Sentence Transformers` (for embedding generation)
  - `PyMuPDF` or `pdfplumber` (for document parsing)

## Key Features / Value Proposition
- **Free and Open-Source**: No reliance on paid APIs, making it a cost-effective solution.
- **Efficient Retrieval**: Uses FAISS to quickly find relevant information from large knowledge bases.
- **Customizable Deployment**: Can be run in Jupyter Notebook, Google Colab, or as a Flask web app.
- **Optimized for CPU**: Designed to run efficiently without requiring a GPU.
- **Enhanced Response Quality**: Combines retrieval with generation to provide accurate and context-aware answers.
- **Scalability**: Easily extendable to support additional document types and retrieval methods.

## Getting Started
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the chatbot:
   ```sh
   flask run
   ```
## Demo

<img width="1470" alt="image" src="https://github.com/user-attachments/assets/9a92d90a-7f15-47e9-9664-d8aba1143cc5" />
