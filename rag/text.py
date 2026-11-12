"""Small text helpers shared by retrieval, generation and grounding."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")
_SUFFIXES = ("ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "s", "ly")


def split_sentences(text):
    """Split text into sentences. Good enough for abstracts and short answers."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def stem(word):
    """Very light suffix stripping so 'hallucinations' matches 'hallucination'."""
    if word[0].isdigit():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_tokens(text):
    """Lower-cased, stemmed words with stop words removed (numbers are kept)."""
    out = []
    for w in _WORD.findall(text.lower()):
        if w in ENGLISH_STOP_WORDS:
            continue
        out.append(stem(w))
    return out


def numbers(text):
    """Numeric tokens such as '71%' or '19.5%', normalised for comparison."""
    return {n.rstrip("%") for n in re.findall(r"\d+(?:\.\d+)?%?", text)}
