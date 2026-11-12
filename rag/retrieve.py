"""Cosine-similarity search over the TF-IDF index."""
import pickle

from sklearn.metrics.pairwise import linear_kernel

from .ingest import INDEX_PATH, build_index

# Chunks scoring below this are treated as "nothing relevant". 0.05 is low enough
# to keep every question in eval/golden_qa.json and high enough to reject
# questions that share no vocabulary with the corpus.
MIN_SCORE = 0.05

_index = None


def load_index(force_rebuild=False):
    """Load the pickled index, building it first if it does not exist yet."""
    global _index
    if _index is None or force_rebuild:
        if force_rebuild or not INDEX_PATH.exists():
            _index = build_index()
        else:
            with open(INDEX_PATH, "rb") as f:
                _index = pickle.load(f)
    return _index


def retrieve(query, top_k=4, min_score=MIN_SCORE):
    """Return up to `top_k` chunks for `query`, best first, each with a `score`."""
    index = load_index()
    scores = linear_kernel(index["vectorizer"].transform([query]), index["matrix"]).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    return [
        {**index["chunks"][i], "score": round(float(scores[i]), 3)}
        for i in ranked
        if scores[i] >= min_score
    ]
