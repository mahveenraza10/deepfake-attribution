"""Check every prerequisite for run_vlm.py in order. Run on a GPU node:

  sbatch slurm/preflight.sbatch          # or, interactively:
  srun --gres=gpu:1 --mem=64G --pty bash
  source ./env.sh && "$EVATTR_VENV/bin/python" preflight.py --root "$FAVC"

Prints PASS/FAIL per step and stops at the first FAIL with the fix.
"""
import argparse, os, sys, glob

def ok(m): print(f"  PASS  {m}")
def bad(m, fix): print(f"  FAIL  {m}\n        fix: {fix}"); sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("FAVC", ""))
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    a = ap.parse_args()

    print("1. python + packages")
    print(f"  python {sys.version.split()[0]}")
    try:
        import torch
        ok(f"torch {torch.__version__}")
    except ImportError:
        bad("torch missing", "pip install torch --index-url https://download.pytorch.org/whl/cu121")
    try:
        import transformers as tf
        v = tuple(int(x) for x in tf.__version__.split(".")[:2])
        if v < (4, 52):
            bad(f"transformers {tf.__version__} too old", "pip install -U 'transformers>=4.52'")
        ok(f"transformers {tf.__version__}")
    except ImportError:
        bad("transformers missing", "pip install 'transformers>=4.52' accelerate")
    try:
        import qwen_omni_utils  # noqa
        ok("qwen_omni_utils")
    except ImportError:
        bad("qwen_omni_utils missing", "pip install 'qwen-omni-utils[decord]'")
    try:
        import decord  # noqa
        ok("decord (video decode)")
    except ImportError:
        print("  WARN  decord missing, falls back to torchvision and is much slower")
    try:
        import soundfile, librosa  # noqa
        ok("soundfile + librosa (audio decode)")
    except ImportError:
        bad("soundfile/librosa missing", "pip install soundfile librosa")

    print("2. GPU")
    if not torch.cuda.is_available():
        bad("no CUDA device visible", "request --gres=gpu:1; do not run on a login node")
    n = torch.cuda.device_count()
    tot = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1e9
    ok(f"{n} GPU(s), {tot:.0f} GB total ({torch.cuda.get_device_name(0)})")
    if "Omni-7B" in a.model and tot < 40:
        bad(f"{tot:.0f} GB is too little for 7B bf16", "use an 80GB A100 or add a second GPU")

    print("3. model class available")
    try:
        if "Qwen3-Omni" in a.model:
            from transformers import Qwen3OmniMoeForConditionalGeneration  # noqa
        else:
            from transformers import Qwen2_5OmniForConditionalGeneration  # noqa
        ok(f"class for {a.model}")
    except ImportError as e:
        bad(f"{e}", "transformers is too old for this model, or install from git: "
                    "pip install git+https://github.com/huggingface/transformers")

    print("4. weights reachable")
    hub = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    local = os.path.isdir(a.model)
    cached = glob.glob(os.path.join(hub, "hub", "models--" + a.model.replace("/", "--")))
    if local:
        ok(f"local path {a.model}")
    elif cached:
        ok(f"cached in {hub}")
    else:
        print(f"  WARN  not cached in {hub}; the node must have internet or you must "
              f"pre-download on a login node:\n        huggingface-cli download {a.model}")

    print("5. dataset")
    if not a.root or not os.path.isdir(a.root):
        bad(f"FakeAVCeleb root not found: {a.root!r}", "set --root / $FAVC to the FakeAVCeleb_v1.2 directory")
    quads = ["RealVideo-RealAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"]
    missing = [q for q in quads if not os.path.isdir(os.path.join(a.root, q))]
    if missing:
        print(f"  FAIL  missing quadrant dirs: {missing}")
        print(f"        contents of {a.root}: {sorted(os.listdir(a.root))[:10]}")
        print("        fix: point --root at the level containing the four quadrant folders, "
              "or use sample.py --manifest with your unified CSV")
        sys.exit(1)
    counts = {q: sum(1 for _ in glob.iglob(os.path.join(a.root, q, "**", "*.mp4"), recursive=True)) for q in quads}
    if min(counts.values()) < 50:
        bad(f"fewer than 50 clips in some quadrant: {counts}", "lower --per-quadrant to the smallest count")
    ok(f"clip counts {counts}")

    print("6. end-to-end on one clip")
    probe = next(glob.iglob(os.path.join(a.root, quads[3], "**", "*.mp4"), recursive=True))
    from run_vlm import load, infer, parse
    from taxonomy import build_prompt
    model, proc = load(a.model)
    ok("model loaded")
    raw = infer(model, proc, probe, build_prompt())
    print(f"  raw output: {raw[:300]}")
    j = parse(raw)
    if j is None:
        bad("output is not parsable JSON", "see raw above; tighten the prompt or raise max_new_tokens")
    ok(f"parsed: verdict={j['verdict']} evidence={[e['category'] for e in j['evidence']]}")
    print("\nALL CHECKS PASSED - sbatch run.sbatch")


if __name__ == "__main__":
    main()
