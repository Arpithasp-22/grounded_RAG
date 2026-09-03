"""
Grounded RAG Assistant — Streamlit demo.

Ask a question about LLM hallucination-detection research. The answer
is generated from retrieved paper excerpts, then every sentence is
checked against those excerpts by a separate, deterministic scorer —
the LLM never gets to grade its own homework.

Run:
    streamlit run app.py
"""
import streamlit as st

from rag.retrieve import retrieve
from rag.generate import generate_answer
from rag.grounding import score_groundedness

st.set_page_config(page_title="Grounded RAG Assistant", page_icon="🔎", layout="centered")

st.title("🔎 Grounded RAG Assistant")
st.caption(
    "Ask a question about LLM hallucination-detection research. "
    "Every sentence of the answer is checked against the retrieved sources — "
    "if it isn't backed by evidence, it's flagged, not hidden."
)

with st.expander("What's in the corpus?"):
    st.write(
        "8 real papers on LLM hallucination detection — SelfCheckGPT, semantic entropy, "
        "FacTool, HaloScope, and others — chunked and indexed locally with TF-IDF. "
        "No vector database or model download needed to run this."
    )

query = st.text_input(
    "Your question",
    placeholder="e.g. How can you detect hallucinations without access to model internals?",
)

if st.button("Ask", type="primary") and query:
    with st.spinner("Retrieving relevant excerpts…"):
        chunks = retrieve(query, top_k=4)

    if not chunks:
        st.warning("Nothing relevant found in the corpus for that question.")
    else:
        with st.spinner("Generating answer…"):
            result = generate_answer(query, chunks, use_llm=True)

        grounding = score_groundedness(result["answer"], chunks)

        st.subheader("Answer")
        for sentence in grounding["sentences"]:
            if sentence["grounded"]:
                st.markdown(f":green[{sentence['text']}]  `✓ {sentence['best_score']}`")
            else:
                st.markdown(f":red[{sentence['text']}]  `✗ ungrounded`")

        col1, col2, col3 = st.columns(3)
        col1.metric("Groundedness", f"{grounding['groundedness_pct']}%")
        col2.metric("Latency", f"{result['latency_ms']} ms")
        col3.metric("Mode", result["mode"].split(" ")[0])

        st.subheader("Retrieved sources")
        for c in chunks:
            st.markdown(f"**{c['title']}** ({c['year']}) — score `{c['score']}`")
            st.caption(c["chunk_text"])
            st.markdown(f"[{c['url']}]({c['url']})")
            st.divider()

st.caption(
    "No OPENAI_API_KEY? The app falls back to a deterministic extractive answer "
    "built straight from the retrieved text — the pipeline never goes fully offline."
)
