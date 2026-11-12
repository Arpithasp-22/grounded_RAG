"""Deterministic groundedness checker. It never calls an LLM.

Each answer sentence is compared with every window of one or two consecutive
sentences from the retrieved chunks. A sentence is *grounded* when

  1. coverage >= threshold, where coverage is the share of the sentence's content
     words (stop words removed, lightly stemmed) that appear in the best
     evidence window, and
  2. every number in the sentence (for example "71%") also appears somewhere in
     the retrieved evidence.

What this can and cannot catch (measured by `python -m eval.run_eval`):
  - It catches off-topic claims and invented numbers.
  - It only partly catches plausible claims that reuse the corpus vocabulary.
  - It is lexical, so it cannot detect a contradiction that reuses the source's
    words, for example a claim with the negation removed.
"""
from .text import content_tokens, numbers, split_sentences

DEFAULT_THRESHOLD = 0.6


def _evidence_windows(chunks):
    """Yield (title, token set, text) for each 1- and 2-sentence window in the chunks."""
    for chunk in chunks:
        sentences = split_sentences(chunk["chunk_text"])
        for i in range(len(sentences)):
            for size in (1, 2):
                window = sentences[i:i + size]
                if len(window) == size:
                    text = " ".join(window)
                    yield chunk["title"], set(content_tokens(text)), text


def score_groundedness(answer, chunks, threshold=DEFAULT_THRESHOLD):
    """Check every sentence of `answer` against `chunks`.

    Returns {"sentences": [...], "groundedness_pct": float}. Each sentence dict has
    text, grounded (bool), best_score (coverage, 0 to 1), source (title of the best
    supporting paper) and reason.
    """
    windows = list(_evidence_windows(chunks))
    evidence_numbers = numbers(" ".join(c["chunk_text"] for c in chunks))

    results = []
    for sentence in split_sentences(answer):
        tokens = set(content_tokens(sentence))
        if not tokens:
            continue  # nothing checkable, e.g. "Yes."

        best_score, best_source = 0.0, None
        for title, window_tokens, _ in windows:
            coverage = len(tokens & window_tokens) / len(tokens)
            if coverage > best_score:
                best_score, best_source = coverage, title

        missing_numbers = sorted(numbers(sentence) - evidence_numbers)
        if missing_numbers:
            grounded, reason = False, f"number not in sources: {', '.join(missing_numbers)}"
        elif best_score >= threshold:
            grounded, reason = True, "supported by a retrieved passage"
        else:
            grounded, reason = False, f"only {best_score:.0%} of its content words appear in the sources"

        results.append({
            "text": sentence,
            "grounded": grounded,
            "best_score": round(best_score, 2),
            "source": best_source if grounded else None,
            "reason": reason,
        })

    grounded_count = sum(r["grounded"] for r in results)
    pct = round(100 * grounded_count / len(results), 1) if results else 0.0
    return {"sentences": results, "groundedness_pct": pct}
