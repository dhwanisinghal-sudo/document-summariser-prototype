# Document Summariser Prototype

Prototype and evaluation of **extractive vs abstractive summarisation** for a content team that condenses long public-policy and news articles into briefs.

## What is in this repo
| File | Purpose |
|---|---|
| `src/preprocess.py` | Cleans articles: removes boilerplate lines, photo captions and duplicate paragraphs |
| `src/extractive.py` | TextRank baseline (TF-IDF or sentence-embedding similarity) |
| `src/abstractive.py` | BART and Pegasus summarisers, chunk-and-merge for long documents, disk cache |
| `src/tune.py` | Grid search of summary length, beam size and chunk overlap on a validation split |
| `src/evaluate.py` | ROUGE-1/2/L, human rating sheet and aggregation, automatic hallucination screen, latency and memory |
| `summariser.ipynb` | Notebook running the full pipeline |
| `run_experiments.py` | Script version of the pipeline |
| `app.py` | Streamlit demo with length, beam and overlap controls |

## Setup
```
pip install -r requirements.txt
```
Place the corpus in `data/articles.jsonl` (see `data/README.md`): 100 articles, 30 with human reference summaries.

## Run
```
python run_experiments.py data/articles.jsonl                # TextRank only
python run_experiments.py data/articles.jsonl --abstractive  # + BART and Pegasus (downloads model weights)
streamlit run app.py                                          # demo
```
Results are written to `results/` (metrics, summaries, tuning grids, human rating sheet).

## Method
1. **Clean** the text, then split 30 reference articles into 10 validation / 20 test.
2. **Baseline:** TextRank over a sentence-similarity graph.
3. **Abstractive:** `facebook/bart-large-cnn` and `google/pegasus-cnn_dailymail`. Articles longer than the token limit are split into sentence-aligned chunks with overlap, each chunk is summarised, and the partial summaries are merged and condensed again if needed.
4. **Tune** length, beam size and chunk overlap on the validation split by ROUGE-L.
5. **Evaluate** on the test split with ROUGE-1/2/L, a 15-sample human rating (faithfulness, coverage, fluency, hallucination), and an automatic check that flags numbers or names in a summary that do not appear in the source.
6. **Stretch:** latency and memory per article on CPU, caching, and cost/speed trade-offs.

## Success criteria
Beat the extractive baseline on ROUGE-L by the agreed margin, keep hallucination under 10% of rated samples, return a summary within 10 seconds.

## Notes
- `data/sample_articles.jsonl` is three invented articles for smoke tests only. It is not the evaluation corpus.
- The automatic hallucination screen is a heuristic to prioritise manual review, not a verdict.
- Model weights are downloaded from Hugging Face on first run.
