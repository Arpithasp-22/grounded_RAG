"""Chunk the corpus and build a local TF-IDF index.

    python -m rag.ingest

No model download, GPU or vector database is needed. The index is a pickle of
a scikit-learn vectorizer, its sparse matrix and the chunk metadata.
"""
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from .text import split_sentences

ROOT = Path(__file__).resolve().parents[1]
PAPERS_PATH = ROOT / "data" / "papers.json"
INDEX_PATH = ROOT / "data" / "index.pkl"

CHUNK_SENTENCES = 3   # sentences per chunk
CHUNK_STRIDE = 2      # a one-sentence overlap between neighbouring chunks


def chunk_sentences(sentences, size=CHUNK_SENTENCES, stride=CHUNK_STRIDE):
    """Sliding windows of `size` sentences; the last window always reaches the end."""
    chunks, i = [], 0
    while sentences:
        chunks.append(sentences[i:i + size])
        if i + size >= len(sentences):
            break
        i += stride
    return chunks


def build_chunks(papers):
    chunks = []
    for paper in papers:
        for n, group in enumerate(chunk_sentences(split_sentences(paper["summary"]))):
            chunks.append({
                "chunk_id": f"{paper['id']}#{n}",
                "paper_id": paper["id"],
                "title": paper["title"],
                "year": paper["year"],
                "url": paper["url"],
                "chunk_text": " ".join(group),
            })
    return chunks


def build_index(papers_path=PAPERS_PATH, index_path=INDEX_PATH):
    papers = json.loads(Path(papers_path).read_text(encoding="utf-8"))
    chunks = build_chunks(papers)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    # The title is indexed with each chunk so a question can match on the paper's name.
    matrix = vectorizer.fit_transform(f"{c['title']}. {c['chunk_text']}" for c in chunks)
    index = {"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(index, f)
    return index


if __name__ == "__main__":
    idx = build_index()
    print(f"Indexed {len(idx['chunks'])} chunks from "
          f"{len({c['paper_id'] for c in idx['chunks']})} papers -> {INDEX_PATH}")
