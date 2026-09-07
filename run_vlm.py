"""Run an audio-visual Qwen model over sample.csv and write one JSON line per clip to preds.jsonl.

Usage:
  python run_vlm.py --sample sample.csv --model Qwen/Qwen2.5-Omni-7B --out preds.jsonl
  python run_vlm.py --sample sample.csv --model Qwen/Qwen3-Omni-30B-A3B-Instruct --out preds.jsonl

Resumable: clips already in --out are skipped. Requires: transformers>=4.52, qwen-omni-utils, torch, flash-attn optional.
"""
import argparse, csv, json, re, time, os
import torch
import transformers
from qwen_omni_utils import process_mm_info
from taxonomy import build_prompt, clip_path, CATEGORIES, SIGNALS

SYS = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

TF_MAJOR = int(transformers.__version__.split(".")[0])


def load(model_name):
    if "Qwen3-Omni" in model_name:
        from transformers import Qwen3OmniMoeForConditionalGeneration as M, Qwen3OmniMoeProcessor as P
    else:
        from transformers import Qwen2_5OmniForConditionalGeneration as M, Qwen2_5OmniProcessor as P
    # transformers v5 renamed from_pretrained's `torch_dtype` to `dtype`. The old
    # name is not an alias there, it lands in **kwargs and is ignored, so the model
    # loads in fp32: 2x the memory and roughly half the throughput.
    dtype_key = "dtype" if TF_MAJOR >= 5 else "torch_dtype"
    kw = {dtype_key: torch.bfloat16, "device_map": "auto"}
    try:
        model = M.from_pretrained(model_name, attn_implementation="flash_attention_2", **kw)
    except Exception:
        model = M.from_pretrained(model_name, **kw)
    # Only worth doing before v5. There, generate() assembles its talker kwargs
    # unconditionally -- it reads self.talker.codec_pad_token before it checks whether
    # audio was even asked for -- so a disabled talker turns every call into an
    # AttributeError. Leaving it loaded costs some idle VRAM and nothing else, since
    # return_audio=False already keeps it out of the forward path.
    if TF_MAJOR < 5 and hasattr(model, "disable_talker"):
        model.disable_talker()
    return model, P.from_pretrained(model_name)


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    v = str(j.get("verdict", "")).lower()
    j["verdict"] = "fake" if "fake" in v or "synth" in v else "real" if "real" in v or "auth" in v else "unk"
    ev = []
    for e in j.get("evidence", []) or []:
        c = str(e.get("category", "")).lower().replace("-", "_").replace(" ", "_")
        if c not in CATEGORIES:
            # recover category from signal name
            s = str(e.get("signal", "")).lower().replace("-", "_").replace(" ", "_")
            c = next((k for k, v in SIGNALS.items() if s in v), None)
        if c:
            ev.append({"category": c, "signal": e.get("signal", ""), "note": e.get("note", "")})
    j["evidence"] = ev
    return j


@torch.no_grad()
def infer(model, proc, path, prompt, max_frames=16, fps=2.0, max_new_tokens=400):
    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYS}]},
        {"role": "user", "content": [{"type": "video", "video": path, "max_frames": max_frames, "fps": fps},
                                     {"type": "text", "text": prompt}]},
    ]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    # The processor needs the fps too, and it must be the rate the frames were ACTUALLY
    # sampled at, which is not the fps requested above. The sampler spreads max_frames
    # linearly over the whole clip, so anything longer than max_frames/fps seconds comes
    # back at a lower effective rate (16 frames over a 15 s clip is ~1.07 fps, not 2).
    #
    # This matters because the processor turns fps into video_second_per_grid, the time
    # grid TMRoPE interleaves the audio against, and it silently defaults to 2.0. Get it
    # wrong and the frames are laid over a different span than the audio occupies, which
    # injects an audio-video desync into every clip, reals included -- and that desync
    # reads as lip_sync_mismatch, the exact cross_modal signal this experiment scores
    # RVFA on. return_video_kwargs gives the true per-clip rate; v5 wants it as a scalar.
    audios, images, videos, vkw = process_mm_info(conv, use_audio_in_video=True,
                                                  return_video_kwargs=True)
    true_fps = vkw.get("fps", fps)
    if isinstance(true_fps, (list, tuple)):
        true_fps = true_fps[0]
    inputs = proc(text=text, audio=audios, images=images, videos=videos, fps=float(true_fps),
                  return_tensors="pt", padding=True, use_audio_in_video=True).to(model.device)
    # thinker_max_new_tokens, not max_new_tokens: generate() seeds the thinker's
    # budget from its own default first and only fills in unprefixed kwargs that are
    # still missing, so a bare max_new_tokens is silently dropped and every clip runs
    # to the 1024-token default. Both model families accept the prefixed name.
    out = model.generate(**inputs, use_audio_in_video=True, return_audio=False,
                         thinker_max_new_tokens=max_new_tokens, do_sample=False)
    if isinstance(out, tuple):
        out = out[0]
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="sample.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--out", default="preds.jsonl")
    ap.add_argument("--max-frames", type=int, default=16)
    # 2.0 is Qwen2.5-Omni's native audio-video rate; max_frames then caps coverage at
    # max_frames/fps seconds of clip. Both values are recorded per row, because the
    # frame budget bounds which temporal and physiological signals are observable at
    # all and so belongs with the numbers it constrains.
    ap.add_argument("--fps", type=float, default=2.0)
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["path"] for l in open(a.out)}
    rows = [r for r in csv.DictReader(open(a.sample)) if r["path"] not in done]
    print(f"{len(rows)} clips to run ({len(done)} already done)")

    model, proc = load(a.model)
    prompt = build_prompt()
    with open(a.out, "a") as f:
        for i, r in enumerate(rows):
            t = time.time()
            try:
                raw = infer(model, proc, clip_path(r["path"]), prompt, a.max_frames, a.fps)
                err = None
            except Exception as e:
                raw, err = "", repr(e)
            j = parse(raw) or {"verdict": "unk", "confidence": None, "evidence": []}
            rec = {**r, "raw": raw, "error": err, **j, "model": a.model,
                   "max_frames": a.max_frames, "fps": a.fps, "sec": round(time.time() - t, 1)}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{i+1}/{len(rows)}] {r['quadrant']} -> {j['verdict']} {[e['category'] for e in j['evidence']]} ({rec['sec']}s)")


if __name__ == "__main__":
    main()
