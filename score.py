"""Score preds.jsonl: prediction accuracy vs attribution correctness (agenda item 1) and evidence sufficiency (item 2).

Metrics
  acc            verdict == ground truth (fake if quadrant != RVRA)
  attr_correct   among true fakes predicted fake: >=1 cited category is in EXPECTED[quadrant]
  attr_precision among cited categories on true fakes: fraction in EXPECTED[quadrant]
  ghost_rate     among true reals: fraction where the model cited any evidence (verdict fake or not)
  n_evidence     mean number of evidence items cited on true fakes
Outputs a markdown + LaTeX table and bootstrap 95% CIs.
"""
import argparse, json, random
from collections import defaultdict, Counter
from taxonomy import EXPECTED, CATEGORIES


def load(p):
    return [json.loads(l) for l in open(p)]


def metrics(rows):
    m = {}
    fakes = [r for r in rows if r["quadrant"] != "RVRA"]
    reals = [r for r in rows if r["quadrant"] == "RVRA"]
    gt = lambda r: "fake" if r["quadrant"] != "RVRA" else "real"
    m["n"] = len(rows)
    m["acc"] = sum(r["verdict"] == gt(r) for r in rows) / max(len(rows), 1)
    tp = [r for r in fakes if r["verdict"] == "fake"]
    m["recall"] = len(tp) / max(len(fakes), 1)
    m["fpr"] = sum(r["verdict"] == "fake" for r in reals) / max(len(reals), 1)
    cats = lambda r: {e["category"] for e in r["evidence"]}
    ok = [len(cats(r) & EXPECTED[r["quadrant"]]) > 0 for r in tp]
    m["attr_correct"] = sum(ok) / max(len(tp), 1)
    cited = [(c in EXPECTED[r["quadrant"]]) for r in fakes for c in cats(r)]
    m["attr_precision"] = sum(cited) / max(len(cited), 1)
    m["ghost_rate"] = sum(len(r["evidence"]) > 0 for r in reals) / max(len(reals), 1)
    m["n_evidence"] = sum(len(r["evidence"]) for r in fakes) / max(len(fakes), 1)
    return m


def boot(rows, key, k=1000, seed=0):
    random.seed(seed)
    vals = sorted(metrics(random.choices(rows, k=len(rows)))[key] for _ in range(k))
    return vals[int(0.025 * k)], vals[int(0.975 * k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds.jsonl")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rows = [r for r in load(a.preds) if r.get("verdict") != "unk"]
    unk = len(load(a.preds)) - len(rows)
    print(f"{len(rows)} scored, {unk} unparsable/unk dropped\n")

    keys = ["acc", "attr_correct", "attr_precision", "ghost_rate", "n_evidence"]
    overall = metrics(rows)
    ci = {k: boot(rows, k, a.boot) for k in keys if k != "n_evidence"}

    print("| Subset | n | Acc | Attr. correct | Attr. precision | Ghost evidence (reals) | Evidence / clip |")
    print("|---|---|---|---|---|---|---|")
    for q in ["ALL", "FVRA", "RVFA", "FVFA", "RVRA"]:
        sub = rows if q == "ALL" else [r for r in rows if r["quadrant"] == q]
        if not sub:
            continue
        m = metrics(sub)
        print(f"| {q} | {m['n']} | {m['acc']:.2f} | {m['attr_correct']:.2f} | {m['attr_precision']:.2f} | "
              f"{m['ghost_rate']:.2f} | {m['n_evidence']:.1f} |")

    print("\n95% bootstrap CI (ALL):")
    for k, (lo, hi) in ci.items():
        print(f"  {k}: {overall[k]:.2f} [{lo:.2f}, {hi:.2f}]")

    # Category confusion: which categories are cited per quadrant
    print("\nCited categories per quadrant (count; * = expected):")
    for q in ["FVRA", "RVFA", "FVFA", "RVRA"]:
        c = Counter(e["category"] for r in rows if r["quadrant"] == q for e in r["evidence"])
        print(f"  {q}: " + ", ".join(f"{k}{'*' if k in EXPECTED[q] else ''}={c[k]}" for k in CATEGORIES))

    # Evidence sufficiency: accuracy conditioned on number of cited categories
    print("\nAccuracy by #cited categories (true fakes):")
    byk = defaultdict(list)
    for r in rows:
        if r["quadrant"] != "RVRA":
            byk[len({e['category'] for e in r['evidence']})].append(r["verdict"] == "fake")
    for k in sorted(byk):
        print(f"  {k} categories: n={len(byk[k])}, recall={sum(byk[k])/len(byk[k]):.2f}")

    # LaTeX
    print("\n% LaTeX")
    print("\\begin{tabular}{lrrrrr}\\toprule\nSubset & $n$ & Acc & Attr.\\ corr. & Attr.\\ prec. & Ghost ev. \\\\\\midrule")
    for q in ["FVRA", "RVFA", "FVFA", "RVRA", "ALL"]:
        sub = rows if q == "ALL" else [r for r in rows if r["quadrant"] == q]
        if not sub:
            continue
        m = metrics(sub)
        g = f"{m['ghost_rate']:.2f}" if q in ("RVRA", "ALL") else "--"
        ac = f"{m['attr_correct']:.2f}" if q != "RVRA" else "--"
        print(f"{q} & {m['n']} & {m['acc']:.2f} & {ac} & {m['attr_precision']:.2f} & {g} \\\\")
    print("\\bottomrule\\end{tabular}")


if __name__ == "__main__":
    main()
