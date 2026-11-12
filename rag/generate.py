"""Answer generation with a deterministic fallback.

If OPENAI_API_KEY is set the answer comes from an OpenAI model, instructed to use
only the retrieved excerpts. If there is no key, or the API call fails for any
reason, the answer is built extractively from the highest-scoring sentences of the
retrieved text, so the app never breaks.
"""
import os
import time

from dotenv import load_dotenv

from .text import content_tokens, split_sentences

load_dotenv()

SYSTEM_PROMPT = (
    "You answer questions about LLM hallucination-detection research. "
    "Use ONLY the numbered excerpts provided. If they do not contain the answer, say so. "
    "Write 2 to 4 plain sentences and do not add facts that are not in the excerpts."
)


RELEVANCE_CUTOFF = 0.5  # use only chunks scoring at least half of the best chunk's score


def extractive_answer(query, chunks, max_sentences=3):
    """Pick the sentences that share the most content words with the query.

    Chunks that matched much more weakly than the best one are ignored, so a
    barely related paper does not leak a sentence into the answer.
    """
    query_tokens = set(content_tokens(query))
    best = max((c.get("score", 1.0) for c in chunks), default=1.0)
    chunks = [c for c in chunks if c.get("score", 1.0) >= RELEVANCE_CUTOFF * best]
    candidates, seen = [], set()
    for rank, chunk in enumerate(chunks):
        for sentence in split_sentences(chunk["chunk_text"]):
            if sentence in seen:
                continue
            seen.add(sentence)
            overlap = len(query_tokens & set(content_tokens(sentence)))
            # Prefer more overlap, then the higher-ranked chunk, then earlier position.
            candidates.append((-overlap, rank, len(candidates), sentence))
    picked = sorted(candidates)[:max_sentences]
    # Present in the order retrieved, which keeps the answer readable.
    return " ".join(s for *_, s in sorted(picked, key=lambda c: (c[1], c[2])))


def _llm_answer(query, chunks, model):
    from openai import OpenAI  # imported here so the fallback needs no OpenAI setup

    context = "\n\n".join(f"[{i}] {c['title']} ({c['year']}): {c['chunk_text']}"
                          for i, c in enumerate(chunks, 1))
    response = OpenAI().chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_answer(query, chunks, use_llm=True):
    """Return {"answer", "latency_ms", "mode"}. `mode` starts with LLM or Extractive."""
    start = time.perf_counter()
    answer, mode = None, "Extractive (no LLM)"

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            answer, mode = _llm_answer(query, chunks, model), f"LLM ({model})"
        except Exception as exc:  # network, auth, rate limit, bad model name...
            mode = f"Extractive (LLM failed: {type(exc).__name__})"

    if answer is None:
        answer = extractive_answer(query, chunks)

    return {
        "answer": answer,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "mode": mode,
    }
