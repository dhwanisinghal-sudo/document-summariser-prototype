"""End-to-end run.
  python run_experiments.py data/articles.jsonl                 # extractive only
  python run_experiments.py data/articles.jsonl --abstractive   # + BART, Pegasus
Expects JSONL with: id, text, reference (reference only for the 30 human-summarised)."""
import json
import sys
from src.preprocess import clean_text
from src.extractive import textrank_summary
from src.evaluate import rouge, flagged_rate, write_rating_sheet, timed, peak_memory_mb

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = args[0] if args else "data/articles.jsonl"
docs = [json.loads(l) for l in open(path, encoding="utf-8")]
for d in docs:
    d["clean"] = clean_text(d["text"])
refd = [d for d in docs if d.get("reference")]
cut = max(1, len(refd) // 3)
val, test = refd[:cut], refd[cut:] or refd[:cut]   # 10 val / 20 test when 30 refs
print(f"{len(docs)} articles, {len(refd)} with references ({len(val)} val / {len(test)} test)")

methods, latency = {}, {}
out = [timed(textrank_summary, d["clean"]) for d in test]
methods["textrank"] = [o[0] for o in out]
latency["textrank"] = round(sum(o[1] for o in out) / len(out), 3)

if "--abstractive" in sys.argv:
    from src.abstractive import Summariser
    from src.tune import grid_search, save
    for name in ("bart", "pegasus"):
        s = Summariser(name)
        best, allr = grid_search(s, [d["clean"] for d in val], [d["reference"] for d in val])
        save(f"results/tuning_{name}.json", best, allr)
        out = [timed(s.summarise, d["clean"], max_len=best["max_len"],
                     min_len=int(best["max_len"] * .3), beams=best["beams"],
                     overlap_sents=best["overlap"], use_cache=False) for d in test]
        methods[name] = [o[0] for o in out]
        latency[name] = round(sum(o[1] for o in out) / len(out), 2)
        print(name, "mean latency (s):", latency[name], "| peak mem MB:", peak_memory_mb())

refs = [d["reference"] for d in test]
srcs = [d["clean"] for d in test]
table = {}
for m, preds in methods.items():
    table[m] = {**rouge(preds, refs), "auto_flagged_%": flagged_rate(preds, srcs),
                "mean_latency_s": latency[m]}
    print(m, table[m])
json.dump(table, open("results/metrics.json", "w"), indent=2)
write_rating_sheet("results/human_rating_sheet.csv", [d["id"] for d in test[:15]],
                   {m: p[:15] for m, p in methods.items()})
json.dump(methods, open("results/summaries.json", "w"), indent=1)
