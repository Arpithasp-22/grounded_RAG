import json
from pathlib import Path

import pytest

from rag import generate as gen
from rag.generate import extractive_answer, generate_answer
from rag.grounding import score_groundedness
from rag.ingest import build_chunks, chunk_sentences, PAPERS_PATH
from rag.retrieve import retrieve
from rag.text import numbers, split_sentences


def chunk(text, title="Paper"):
    return {"chunk_text": text, "title": title}


# ---- ingest ---------------------------------------------------------------

def test_chunk_windows_cover_every_sentence_and_end_at_the_last():
    sentences = [f"Sentence {i}." for i in range(7)]
    chunks = chunk_sentences(sentences)
    assert set(sum(chunks, [])) == set(sentences)
    assert chunks[-1][-1] == sentences[-1]


def test_every_paper_has_chunks_and_metadata():
    papers = json.loads(Path(PAPERS_PATH).read_text())
    chunks = build_chunks(papers)
    assert {c["paper_id"] for c in chunks} == {p["id"] for p in papers}
    assert all(c["url"].startswith("https://arxiv.org/abs/") for c in chunks)


# ---- retrieve -------------------------------------------------------------

def test_retrieval_finds_the_right_paper_first():
    top = retrieve("What is semantic entropy?")[0]
    assert top["paper_id"] == "semantic_uncertainty"


def test_retrieval_scores_are_sorted_and_respect_top_k():
    results = retrieve("hallucination detection with sampled responses", top_k=3)
    scores = [r["score"] for r in results]
    assert len(results) <= 3 and scores == sorted(scores, reverse=True)


def test_vocabulary_free_question_returns_nothing():
    assert retrieve("How do I bake sourdough bread?") == []


# ---- grounding ------------------------------------------------------------

EVIDENCE = [chunk("HaluEval is a benchmark for testing language models. "
                  "About 19.5% of ChatGPT responses contained hallucinations.")]


def test_sentence_copied_from_evidence_is_grounded():
    out = score_groundedness("HaluEval is a benchmark for testing language models.", EVIDENCE)
    assert out["sentences"][0]["grounded"] and out["groundedness_pct"] == 100.0


def test_invented_number_is_flagged_even_with_matching_words():
    out = score_groundedness("About 45% of ChatGPT responses contained hallucinations.", EVIDENCE)
    s = out["sentences"][0]
    assert not s["grounded"] and "45" in s["reason"]


def test_off_topic_sentence_is_flagged():
    out = score_groundedness("The recipe needs two cups of flour.", EVIDENCE)
    assert not out["sentences"][0]["grounded"]


def test_groundedness_percentage_counts_sentences():
    answer = "HaluEval is a benchmark for testing language models. The moon is made of cheese."
    assert score_groundedness(answer, EVIDENCE)["groundedness_pct"] == 50.0


def test_empty_and_stopword_only_answers_do_not_crash():
    assert score_groundedness("", EVIDENCE)["groundedness_pct"] == 0.0
    assert score_groundedness("It is what it is.", EVIDENCE)["sentences"] == []


def test_known_limit_negation_removed_claim_is_not_caught():
    """Documents a limitation: a lexical checker cannot see a dropped negation."""
    evidence = [chunk("The method works without access to internal probabilities.")]
    out = score_groundedness("The method works with access to internal probabilities.", evidence)
    assert out["sentences"][0]["grounded"]


# ---- generate -------------------------------------------------------------

def test_without_a_key_the_answer_is_extractive_and_grounded(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chunks = retrieve("How does Chain-of-Verification reduce hallucination?")
    result = generate_answer("How does Chain-of-Verification reduce hallucination?", chunks)
    assert result["mode"].startswith("Extractive")
    assert score_groundedness(result["answer"], chunks)["groundedness_pct"] == 100.0


def test_llm_failure_falls_back_instead_of_raising(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(gen, "_llm_answer", boom)
    chunks = retrieve("What is semantic entropy?")
    result = generate_answer("What is semantic entropy?", chunks)
    assert result["mode"].startswith("Extractive") and "RuntimeError" in result["mode"]
    assert result["answer"]


def test_llm_answer_is_used_when_available(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(gen, "_llm_answer", lambda q, c, m: "A stubbed answer.")
    result = generate_answer("What is semantic entropy?", retrieve("What is semantic entropy?"))
    assert result["answer"] == "A stubbed answer." and result["mode"].startswith("LLM")


# ---- text helpers ---------------------------------------------------------

def test_sentence_split_keeps_decimals_together():
    assert len(split_sentences("About 19.5% were wrong. The next one is fine.")) == 2


def test_numbers_are_normalised():
    assert numbers("71% to 83% and 19.5%") == {"71", "83", "19.5"}
