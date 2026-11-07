"""Streamlit demo: paste an article, pick a method, control length and chunking.
Run: streamlit run app.py"""
import time
import streamlit as st
from src.preprocess import clean_text
from src.extractive import textrank_summary
from src.evaluate import hallucination_flags

st.set_page_config(page_title="Document Summariser", layout="wide")
st.title("Document Summariser Prototype")
method = st.sidebar.selectbox("Method", ["TextRank (extractive)", "BART (abstractive)",
                                         "Pegasus (abstractive)"])
max_len = st.sidebar.slider("Summary length", 50, 250, 130, 10)
beams = st.sidebar.slider("Beam size (abstractive)", 1, 6, 4)
overlap = st.sidebar.slider("Chunk overlap, sentences (abstractive)", 0, 3, 1)
text = st.text_area("Paste article text", height=300)


@st.cache_resource
def load(name):
    from src.abstractive import Summariser
    return Summariser(name)


if st.button("Summarise") and text.strip():
    clean = clean_text(text)
    t = time.perf_counter()
    if method.startswith("TextRank"):
        out = textrank_summary(clean, max_words=max_len)
    else:
        out = load(method.split()[0].lower()).summarise(
            clean, max_len=max_len, min_len=int(max_len * .3), beams=beams,
            overlap_sents=overlap)
    st.subheader("Summary")
    st.write(out)
    st.caption(f"{len(clean.split())} words in, {len(out.split())} words out, "
               f"{time.perf_counter() - t:.1f}s")
    f = hallucination_flags(out, clean)
    if f["flagged"]:
        st.warning(f"Check against the source: numbers {f['numbers']}, names {f['entities']}")
