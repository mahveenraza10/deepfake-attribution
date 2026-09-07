"""Sample a balanced set of FakeAVCeleb clips (N per quadrant) into sample.csv.

Usage:
  python sample.py --root /path/to/FakeAVCeleb_v1.2 --per-quadrant 50
  python sample.py --manifest test.csv --path-col path --quadrant-col quadrant --per-quadrant 50

Raw layout expected: <root>/<Quadrant>/<race>/<gender>/<id>/*.mp4

Paths are written relative to --root. The `method` column is parsed from the filename
and is INDICATIVE ONLY: fsgan clips carry no method token and come out as "none". Use
by_method.py, which reads the dataset's own meta_data.csv, for anything load-bearing.
"""
import argparse, csv, random, re
from pathlib import Path
from taxonomy import QUADRANT_DIRS

METHOD_RE = re.compile(r"(faceswap|fsgan|wav2lip|wavtolip|rtvc)", re.I)


def from_root(root):
    """Paths are stored RELATIVE to root, so a sample.csv (and the predictions
    derived from it) remains valid on any machine that has the dataset."""
    rows = []
    for d, q in QUADRANT_DIRS.items():
        for p in (Path(root) / d).rglob("*.mp4"):
            methods = sorted({m.lower().replace("wavtolip", "wav2lip") for m in METHOD_RE.findall(p.stem)})
            rows.append({"path": str(p.relative_to(root)), "quadrant": q,
                         "method": "+".join(methods) or "none", "identity": p.parent.name})
    return rows


def from_manifest(path, path_col, quad_col, method_col):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            q = r[quad_col]
            q = QUADRANT_DIRS.get(q, q)
            rows.append({"path": r[path_col], "quadrant": q,
                         "method": r.get(method_col, "") or "none", "identity": r.get("identity", "")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--manifest")
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--quadrant-col", default="quadrant")
    ap.add_argument("--method-col", default="method")
    ap.add_argument("--per-quadrant", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sample.csv")
    a = ap.parse_args()

    rows = from_root(a.root) if a.root else from_manifest(a.manifest, a.path_col, a.quadrant_col, a.method_col)
    random.seed(a.seed)
    out = []
    for q in ["RVRA", "FVRA", "RVFA", "FVFA"]:
        pool = [r for r in rows if r["quadrant"] == q]
        random.shuffle(pool)
        # spread across identities where possible
        seen, pick = set(), []
        for r in pool:
            if r["identity"] in seen:
                continue
            seen.add(r["identity"]); pick.append(r)
            if len(pick) == a.per_quadrant:
                break
        if len(pick) < a.per_quadrant:
            pick += [r for r in pool if r not in pick][: a.per_quadrant - len(pick)]
        print(f"{q}: {len(pick)} / {len(pool)} available")
        out += pick
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "quadrant", "method", "identity"])
        w.writeheader(); w.writerows(out)
    print(f"wrote {len(out)} rows -> {a.out}")


if __name__ == "__main__":
    main()
