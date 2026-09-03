# Grounded RAG Assistant

A retrieval-augmented question-answering system for LLM hallucination-detection
research, with a built-in, deterministic groundedness checker — the LLM is
allowed to write the answer, but a separate, non-LLM scorer decides whether
each sentence of that answer is actually supported by the retrieved evidence.

## Why this project

Every other project in this portfolio makes the same bet: don't trust a
system just because it sounds confident — measure it. This one applies that
directly to the failure mode people care about most right now with LLMs:
hallucination. It's a full RAG pipeline (ingest → retrieve → generate →
verify), it runs entirely for free with no vector database or API key
required, and it evaluates itself against a golden question set the same way
the [LLM Evaluation Framework](../Automated-LLM-Evaluation-Pipeline) does.

## How it works

```
data/papers.json          8 real papers on LLM hallucination detection
        │
        ▼
rag/ingest.py              chunk + TF-IDF index (local, no model download)
        │
        ▼
rag/retrieve.py             cosine similarity search over the index
        │
        ▼
rag/generate.py             LLM answer (if OPENAI_API_KEY set)
        │                   → falls back to a deterministic extractive
        │                     answer built straight from retrieved text
        ▼
rag/grounding.py            splits the answer into sentences and checks
        │                   each one against the retrieved evidence —
        │                   this step never uses an LLM
        ▼
app.py                      Streamlit UI: answer, per-sentence grounding,
                             sources, latency
```

## Design decisions

| Decision | Reason |
|---|---|
| TF-IDF instead of neural embeddings | Runs anywhere with no model download, no GPU, no vector DB service — fully reproducible from a fresh clone |
| Groundedness scored by a separate, non-LLM step | The model that wrote the answer shouldn't be the one grading it |
| LLM call has a deterministic fallback | The app is never broken just because a key is missing or an API call fails |
| Evaluated against a golden QA set | Retrieval hit-rate, groundedness %, and latency are numbers, not vibes |

## Setup

```bash
pip install -r requirements.txt
python -m rag.ingest        # builds the local index from data/papers.json
```

## Run the demo

```bash
streamlit run app.py
```

Optional — for LLM-generated answers instead of the extractive fallback,
copy `.env.example` to `.env` and add an `OPENAI_API_KEY`.

## Run the evaluation

```bash
python -m eval.run_eval
```

Reports retrieval hit-rate, average groundedness %, and average latency
across a golden question set (`eval/golden_qa.json`).

## Project structure

```
grounded_rag_assistant/
├── data/
│   ├── papers.json        # source corpus
│   └── index.pkl          # built by rag/ingest.py (git-ignored)
├── rag/
│   ├── ingest.py
│   ├── retrieve.py
│   ├── generate.py
│   └── grounding.py
├── eval/
│   ├── golden_qa.json
│   └── run_eval.py
├── app.py
├── requirements.txt
└── .env.example
```
