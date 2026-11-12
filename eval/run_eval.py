"""Evaluate retrieval, answer generation and the groundedness checker.

    python -m eval.run_eval
    python -m eval.run_eval --json     # machine-readable output

Three separate measurements:

  1. Retrieval: for each golden question, is the expected paper retrieved
     (hit@1, hit@k, MRR), and are off-topic questions rejected?
  2. Extractive answers: groundedness of the fallback answers. These copy
     sentences from the sources, so ~100% is expected; this is a sanity check,
     not a quality result.
  3. Groundedness checker: labelled faithful and fabricated sentences (hand-written,
     see eval/grounding_cases.json). This is the measurement that says whether the
     checker can tell supported claims from unsupported ones.

The threshold is fixed in advance (rag.grounding.DEFAULT_THRESHOLD). The sweep at
the end is reported for transparency and was not used to choose it.
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from rag.generate import generate_answer
from rag.grounding import DEFAULT_THRESHOLD, score_groundedness
from rag.retrieve import retrieve

HERE = Path(__file__).resolve().parent
TOP_K = 4


def eval_retrieval():
    qa = json.loads((HERE / "golden_qa.json").read_text())
    answerable = [q for q in qa if q["expected_paper"]]
    off_topic = [q for q in qa if not q["expected_paper"]]

    ranks, grounded, pipeline_ms = [], [], []
    for q in answerable:
        t0 = time.perf_counter()
        chunks = retrieve(q["question"], top_k=TOP_K)
        papers = []
        for c in chunks:
            if c["paper_id"] not in papers:
                papers.append(c["paper_id"])
        ranks.append(papers.index(q["expected_paper"]) + 1 if q["expected_paper"] in papers else None)
        if chunks:
            result = generate_answer(q["question"], chunks, use_llm=False)
            grounded.append(score_groundedness(result["answer"], chunks)["groundedness_pct"])
        pipeline_ms.append((time.perf_counter() - t0) * 1000)

    rejected = sum(1 for q in off_topic if not retrieve(q["question"], top_k=TOP_K))
    n = len(answerable)
    return {
        "answerable_questions": n,
        "hit_at_1": sum(r == 1 for r in ranks) / n,
        f"hit_at_{TOP_K}": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "off_topic_questions": len(off_topic),
        "off_topic_rejected": rejected / len(off_topic),
        "extractive_groundedness_pct": round(statistics.mean(grounded), 1),
        "pipeline_ms_mean": round(statistics.mean(pipeline_ms), 1),
        "pipeline_ms_p95": round(sorted(pipeline_ms)[int(0.95 * (len(pipeline_ms) - 1))], 1),
        "misses": [q["question"] for q, r in zip(answerable, ranks) if r != 1],
    }


def eval_grounding(threshold=DEFAULT_THRESHOLD):
    cases = json.loads((HERE / "grounding_cases.json").read_text())
    by_type = defaultdict(lambda: [0, 0])  # type -> [flagged, total] for fabricated
    faithful_ok = faithful_total = 0
    errors = []
    for case in cases:
        chunks = retrieve(case["question"], top_k=TOP_K)
        sentence = score_groundedness(case["sentence"], chunks, threshold)["sentences"][0]
        if case["label"] == "faithful":
            faithful_total += 1
            faithful_ok += sentence["grounded"]
            if not sentence["grounded"]:
                errors.append(("false flag", case["sentence"], sentence["best_score"]))
        else:
            by_type[case["type"]][1] += 1
            by_type[case["type"]][0] += (not sentence["grounded"])
            if sentence["grounded"]:
                errors.append(("missed " + case["type"], case["sentence"], sentence["best_score"]))
    flagged = sum(v[0] for v in by_type.values())
    fabricated_total = sum(v[1] for v in by_type.values())
    return {
        "threshold": threshold,
        "faithful_total": faithful_total,
        "faithful_pass_rate": faithful_ok / faithful_total,
        "fabricated_total": fabricated_total,
        "fabricated_flag_rate": flagged / fabricated_total,
        "fabricated_flag_rate_by_type": {t: f"{v[0]}/{v[1]}" for t, v in sorted(by_type.items())},
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args()

    retrieval = eval_retrieval()
    grounding = eval_grounding()
    sweep = {t: eval_grounding(t) for t in (0.4, 0.5, 0.6, 0.7, 0.8)}

    if args.json:
        print(json.dumps({"retrieval": retrieval, "grounding": grounding,
                          "sweep": {t: {k: v for k, v in r.items() if k.endswith("rate")} for t, r in sweep.items()}},
                         indent=2))
        return

    r, g = retrieval, grounding
    print(f"\n=== Retrieval ({r['answerable_questions']} answerable questions, top_k={TOP_K}) ===")
    print(f"hit@1                    : {r['hit_at_1']:.1%}")
    print(f"hit@{TOP_K}                    : {r[f'hit_at_{TOP_K}']:.1%}")
    print(f"MRR                      : {r['mrr']:.3f}")
    print(f"off-topic rejected       : {r['off_topic_rejected']:.0%} of {r['off_topic_questions']}")
    for q in r["misses"]:
        print(f"  not ranked first: {q}")

    print("\n=== Extractive answers (sanity check; grounded by construction) ===")
    print(f"mean groundedness        : {r['extractive_groundedness_pct']}%")
    print(f"retrieve+answer+check    : {r['pipeline_ms_mean']} ms mean, {r['pipeline_ms_p95']} ms p95 (no LLM)")

    print(f"\n=== Groundedness checker (threshold {g['threshold']}) ===")
    print(f"faithful sentences kept  : {g['faithful_pass_rate']:.1%} of {g['faithful_total']}")
    print(f"fabricated sentences flagged: {g['fabricated_flag_rate']:.1%} of {g['fabricated_total']}")
    for t, v in g["fabricated_flag_rate_by_type"].items():
        print(f"  {t:<22}: {v}")
    print("mistakes:")
    for kind, text, score in g["errors"]:
        print(f"  [{kind}] ({score}) {text}")

    print("\nThreshold sweep (reported, not used to choose the threshold):")
    print("  threshold  faithful kept  fabricated flagged")
    for t, res in sweep.items():
        print(f"  {t:<9}  {res['faithful_pass_rate']:<13.1%}  {res['fabricated_flag_rate']:.1%}")


if __name__ == "__main__":
    main()
