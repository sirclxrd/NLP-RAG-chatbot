# NLP RAG Chatbot  README

Direct and to the point: this repository contains a Retrieval-Augmented Generation
(RAG) system designed to answer questions strictly about Natural Language Processing
(NLP). The knowledge base is built from PDF documents stored in `app_pdf/` (course
notes), indexed with embeddings and FAISS, and queried by a conversational chain that
uses a local Ollama LLM.

Quick start (3 steps)
1. Put your PDF documents into the `app_pdf/` folder.
2. Start the required Ollama models (LLM + embedding model).
3. Run the chatbot: `python RAG.py` and interact via the Gradio web UI.

At-a-glance architecture
- Input: textual question (from web UI or API) + optional conversation history.
- Process: PDF loader  semantic chunking  embeddings  FAISS retriever 
  ConversationalRetrievalChain (LangChain)  Ollama LLM.
- Output: textual answer generated only from retrieved context. If the relevant
  information is not in the context, the system will reply it is not present.

Why this project is conservative
- The system is intentionally conservative to avoid hallucinations: the prompt
  instructs the model to answer only if the information exists in the retrieved
  context. If it cannot be found, the model replies that it does not know.

Requirements
- Python 3.9+ (3.10/3.11 recommended)
- Ollama (local) installed and configured
- Key Python packages (example):

```powershell
pip install gradio langchain langchain-ollama langchain-community \
  langchain-huggingface faiss-cpu transformers numpy
```

Note: on Windows/conda, FAISS installation may require a different procedure.
Use `faiss-cpu` for CPU-only installs or `faiss-gpu` if you have and want GPU support.

Ollama models used
- LLM: `mistral` (or another compatible model). Start with:

```powershell
ollama run mistral
```

- Embedding model: `nomic-embed-text` (or another compatible embedder). Start with:

```powershell
ollama run nomic-embed-text
```

These make models available locally to the code in `RAG.py`.

How documents are processed
- PDFs in `app_pdf/` are loaded with `PyPDFDirectoryLoader`.
- Documents are chunked using `SemanticChunker`.
- Chunks are embedded with `OllamaEmbeddings` and indexed with FAISS.
- The retriever returns top-k hits (k=3 by default) used by the conversational chain.

Run the system (quick commands)

```powershell
# Ensure Ollama + models are running
python RAG.py
```

This starts a Gradio interface where you can ask questions and get context-aware
responses.

Behavior contract (inputs/outputs)
- Input: `question` string and optional in-memory conversation history.
- Output: `answer` string. The chain also returns source documents when enabled.

Common issues and debugging
- FAISS import fails on Windows: try `pip install faiss-cpu` or use conda:
  `conda install -c conda-forge faiss-cpu`.
- Ollama issues: verify model status with `ollama ps` and start models with `ollama run`.
- LangChain API errors: the repo uses `langchain_community` and `langchain_ollama`. If
  you encounter API mismatches, check installed package versions.

Smoke test
1. Start Ollama with the two models.
2. Run this quick import check:

```powershell
python -c "import gradio, langchain, langchain_ollama; print('imports OK')"
```

If OK, run `python RAG.py` and test with a question you know is present in the PDFs.

Suggestions for improvements
- Add a CLI flag to force rebuild the FAISS index only when documents change.
- Save generated answers with source citations to a log for easier QA.
- Add a REST endpoint in addition to Gradio for programmatic access.
- Add unit/integration tests that validate expected answers for a small set of queries.

Contribute
- Create a branch, implement changes (loader, prompt template, or UI), and open a PR.
- If you want, I can create a PR for this README change instead of pushing to `main`.

License
- See `LICENSE` in repository root.

If you want this English README committed to the repo, I can commit and push it now (or
create a PR instead). Which do you prefer?
