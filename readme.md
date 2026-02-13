
# This chatbot
This chatbot was developed as a university course project by a team of four people (including myself).
The main goal was to create a chatbot capable of answering questions exclusively related to Natural Language Processing (NLP), by implementing a RAG (Retrieval-Augmented Generation) system.
# How to: 
# Chatbot RAG per NLP — README

Breve e diretto: questo repository contiene un sistema RAG (Retrieval-Augmented
Generation) pensato per rispondere esclusivamente a domande sul campo del Natural
Language Processing (NLP). Il knowledge base è costruito a partire da documenti PDF
(`app_pdf/`) (appunti del corso), indicizzati con embeddings e FAISS e interrogati
da una catena conversazionale che usa un LLM locale (Ollama).

Per chi vuole andare subito al sodo:
- prepara i documenti PDF in `app_pdf/`;
- avvia i modelli Ollama richiesti;
- lancia `python RAG.py` e interagisci via interfaccia Gradio.

Punti chiave (quick summary)
- Input: domanda testuale dall'interfaccia (o API) + history di conversazione;
- Output: risposta testuale generata SOLO dalle informazioni presenti nel contesto
	recuperato (se l'informazione non è presente, il bot risponde che non lo sa);
- Architettura: loader PDF → semantic chunking → embeddings → FAISS retriever →
	ConversationalRetrievalChain (LangChain) + Ollama LLM.

Requisiti (software)
- Python 3.9+ (consigliato 3.10/3.11)
- Ollama (locale) installato e configurato
- dipendenze Python (esempi):

```powershell
pip install gradio langchain langchain-ollama langchain-community \
	langchain-huggingface faiss-cpu transformers numpy
```

Nota: `faiss-cpu` su Windows/conda può richiedere una procedura diversa. Se hai una
GPU e preferisci usare `faiss-gpu`, segui le istruzioni ufficiali di FAISS.

Modelli Ollama richiesti
- LLM: `mistral` (o altro modello compatibile) — avviare con:

```powershell
ollama run mistral
```

- Embedding model: `nomic-embed-text` (o altro compatibile) — avviare con:

```powershell
ollama run nomic-embed-text
```

Questi comandi rendono i modelli accessibili localmente via Ollama e permettono a
`RAG.py` di creare embeddings e generare risposte senza dipendere da servizi esterni.

Configurazione dei documenti
- Inserisci i PDF (appunti, dispense) in `app_pdf/`.
- Il loader usato è `PyPDFDirectoryLoader`; i documenti vengono splittati tramite
	`SemanticChunker` e indicizzati con embeddings (OllamaEmbeddings nel codice).

Eseguire il sistema (quick start)
1. Assicurati che Ollama e i due modelli siano in esecuzione (vedi sopra).
2. Verifica le dipendenze Python installate.
3. Avvia il chatbot:

```powershell
python RAG.py
```

Questo avvierà un'interfaccia Gradio (web UI) che ti permette di porre domande e
ottenere risposte contestualizzate.

Design & comportamento dell'agente
- Il prompt system (definito in `RAG.py`) impone: "Rispondi SOLO se l'informazione è
	presente nel contesto; altrimenti rispondi che non è presente". Il comportamento è
	volutamente conservativo per evitare hallucination.
- La catena mantiene una memoria conversazionale di finestra (`ConversationBufferWindowMemory`),
	usata per contestualizzare follow-up.

Contract (inputs / outputs)
- Input: stringa `question` (testo), opzionale `chat_history` (lista messaggi);
- Output: stringa di risposta (testo) e — internamente — lista di documenti sorgente
	(se `return_source_documents=True`).

Esempi di prompt
- Utente: "Qual è la differenza tra BERT e GPT?"
- Sistema: recupera chunk rilevanti dai PDF e passa il contesto al LLM; LLM risponde
	solamente con informazioni trovate nei chunk.

Edge cases e limitazioni
- Se il knowledge base non contiene l'argomento, il bot risponderà "not in context";
	questo è voluto ma significa che la copertura dipende totalmente dai documenti forniti.
- Qualità delle risposte dipende da: qualità dei PDF, accuratezza dello splitting,
	e capacità del modello LLM scelto.

Debug / problemi comuni
- Errore FAISS: se l'import di FAISS fallisce su Windows, prova `pip install faiss-cpu`
	o installa FAISS tramite conda (`conda install -c conda-forge faiss-cpu`).
- Ollama: assicurati che i modelli siano caricati correttamente (`ollama ps` / `ollama run ...`).
- Dipendenze LangChain: la repo usa componenti `langchain_community` e `langchain_ollama` —
	verifica le versioni se incontri errori di API.

Test rapido (smoke-test)
1. Avvia Ollama per entrambi i modelli.
2. Esegui nel repo:

```powershell
python -c "import gradio, langchain, langchain_ollama; print('imports ok')"
```

Se OK, avvia `python RAG.py` e prova una domanda banale contenuta nei tuoi PDF.

Suggerimenti di miglioramento
- Salvare l'indice FAISS su disco (già implementato) e aggiungere un flag per forzare
	il rebuild solo quando i documenti cambiano.
- Esporre una route REST oltre all'interfaccia Gradio per integrazione programmatica.
- Aggiungere test automatici che verificano che per un set di query note il sistema
	ritorni risposte attese (gold answers) o segnali correttamente "not in context".

Contribuire
- Se vuoi migliorare: apri una branch, modifica i loader o i template prompt e manda
	una PR. In caso di dubbi, inviami gli errori prodotti e ti aiuto a sistemarli.

License
- Vedi file `LICENSE` nella root del repository.

Se vuoi, applico questa versione del README direttamente nel repo (commit + push),
oppure la traduco in inglese. Dimmi come preferisci procedere.
