"""Re-score preds.jsonl against the dataset's own method labels instead of the
quadrant alone.

sample.py infers the manipulation from the filename, which silently misses fsgan
(its clips carry no method token) and so labels 41% of FakeVideo-RealAudio "none".
meta_data.csv ships the real label for every clip. Nothing here changes EXPECTED --
that mapping is a claim about which evidence is admissible, and belongs to the paper.
This only reports what the numbers look like broken out by true method, so the
question of whether EXPECTED is right for wav2lip can be settled with evidence.

  python by_method.py --preds runs/preds_<model>.jsonl
"""
import argparse, csv, json, os
from collections import Counter, defaultdict
from taxonomy import EXPECTED, CATEGORIES

ROOT = os.environ.get("FAVC", "")


def method_index(root):
    """clip path relative to the dataset root -> method label, from meta_data.csv."""
    idx = {}
    with open(os.path.join(root, "meta_data.csv")) as f:
        for r in csv.DictReader(f):
            # the unnamed trailing column holds the directory, prefixed "FakeAVCeleb/"
            d = (r.get("") or "").strip().lstrip("/")
            if d.startswith("FakeAVCeleb/"):
                d = d[len("FakeAVCeleb/"):]
            idx[os.path.join(d, r["path"].strip())] = r["method"].strip()
    return idx


def rel(p, root):
    """Predictions may carry absolute paths if an external manifest was used."""
    return os.path.relpath(p, root) if os.path.isabs(p) else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--root", default=ROOT, help="FakeAVCeleb root (default: $FAVC)")
    a = ap.parse_args()
    if not a.root:
        ap.error("set --root or $FAVC to the FakeAVCeleb root")

    idx = method_index(a.root)
    rows = [json.loads(l) for l in open(a.preds) if l.strip()]
    rows = [r for r in rows if r.get("verdict") != "unk"]

    hit = sum(rel(r["path"], a.root) in idx for r in rows)
    print(f"{len(rows)} rows, {hit} matched to meta_data.csv\n")
    for r in rows:
        r["true_method"] = idx.get(rel(r["path"], a.root), "UNMATCHED")

    print("filename-parsed method vs dataset label:")
    mism = Counter((r["method"], r["true_method"]) for r in rows)
    for (parsed, true), n in sorted(mism.items(), key=lambda kv: -kv[1]):
        flag = "" if parsed == true or (parsed == "none" and true == "real") else "   <-- mislabelled"
        print(f"  parsed={parsed:18s} true={true:18s} n={n}{flag}")

    print("\nby true method (fakes only):")
    print("| method | quadrant | n | detected | attr. correct | cited categories |")
    print("|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in rows:
        if r["quadrant"] != "RVRA":
            by[(r["true_method"], r["quadrant"])].append(r)
    for (m, q), sub in sorted(by.items()):
        tp = [r for r in sub if r["verdict"] == "fake"]
        cats = lambda r: {e["category"] for e in r["evidence"]}
        ok = sum(len(cats(r) & EXPECTED[q]) > 0 for r in tp)
        cited = Counter(c for r in tp for c in cats(r))
        shown = ", ".join(f"{c}{'*' if c in EXPECTED[q] else '!'}={cited[c]}"
                          for c in CATEGORIES if cited[c])
        ac = f"{ok}/{len(tp)}" if tp else "--"
        print(f"| {m} | {q} | {len(sub)} | {len(tp)} | {ac} | {shown or '--'} |")
    print("\n* = admissible under EXPECTED[quadrant];  ! = inadmissible")


if __name__ == "__main__":
    main()
