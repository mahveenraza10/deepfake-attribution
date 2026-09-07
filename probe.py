"""Perception control: does the model actually see the frames and hear the audio?
If it can describe a clip accurately, a near-constant 'real' verdict is the model's
judgement, not the harness starving it of input."""
import argparse, csv, os
from run_vlm import load, infer
from taxonomy import clip_path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--sample", default=os.path.join(os.environ.get("EVATTR_OUT", "."), "sample.csv"))
ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
ap.add_argument("--per-quadrant", type=int, default=2)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.sample)))
picks = []
for q in ["RVRA", "FVRA", "RVFA", "FVFA"]:
    picks += [r for r in rows if r["quadrant"] == q][:args.per_quadrant]

model, proc = load(args.model)

PROBES = {
 "describe": "Describe this video in two sentences: what the person looks like, and what you can hear.",
 "transcribe": "Transcribe exactly what the speaker says in the audio. Output only the words.",
 "mouth": "Look only at the mouth region across the frames. Describe how the lips move and whether the "
          "mouth area looks like the rest of the face in sharpness and colour. Two sentences.",
 "forced": "Is this talking-head video authentic, or was it synthetically generated or manipulated in any "
           "way (face swap, lip sync, or cloned voice)? You must commit. Answer with one word, real or fake, "
           "then one sentence of justification.",
}
for r in picks:
    print("=" * 100)
    print(f"{r['quadrant']}  method={r['method']}  {r['path'].split('/')[-1]}")
    for name, p in PROBES.items():
        out = infer(model, proc, clip_path(r["path"]), p, max_frames=16, fps=2.0, max_new_tokens=160)
        print(f"  [{name:10s}] {out.strip()[:400]}")
