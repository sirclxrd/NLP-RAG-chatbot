# NLP RAG Chatbot  README

This repository contains a Retrieval-Augmented Generation
(RAG) system designed to answer questions strictly about Natural Language Processing
(NLP). The knowledge base is built from PDF documents stored in `app_pdf/` (course
notes), indexed with embeddings and FAISS, and queried by a conversational chain that
uses a local Ollama LLM.

Quick start (3 steps)
1. Put your PDF documents into the `app_pdf/` folder.
2. Start the required Ollama models (LLM + embedding model).
3. Run the chatbot: `python RAG.py` and interact via the Gradio web UI.
This starts a Gradio interface where you can ask questions and get context-aware responses.

Requirements
- Python 3.9+ (3.10/3.11 recommended)
- Ollama (local) installed and configured
- Key Python packages (example):

```
pip install gradio langchain langchain-ollama langchain-community \
  langchain-huggingface faiss-cpu transformers numpy
```

Ollama models used
- LLM: `mistral` (or another compatible model). Start with:

```
ollama pull mistral
```

- Embedding model: `nomic-embed-text` (or another compatible embedder). Start with:

```
ollama pul nomic-embed-text
```

These make models available locally to the code in `RAG.py`.

How documents are processed
- PDFs in `app_pdf/` are loaded with `PyPDFDirectoryLoader`.
- Documents are chunked using `SemanticChunker`.
- Chunks are embedded with `OllamaEmbeddings` and indexed with FAISS.
- The retriever returns top-k hits (k=3 by default) used by the conversational chain.

## Customizing the prompt

You can easily adapt the system to your own document collection by editing the
prompt template directly in `RAG.py`. Make the prompt coherent with the style and
terminology of your custom documents so the LLM's answers stay aligned with the
knowledge base. 

## Documentation

Implementation details and step-by-step instructions are provided
in `Report.pdf` included in this repository — consult that file for concrete examples and recommended prompt templates.





