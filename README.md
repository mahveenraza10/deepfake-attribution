# Attribution correctness on FakeAVCeleb

Mahveen Raza, Independent Researcher, Toronto, Ontario, Canada 
Syed Raza Bashir, Scientific Research-AI Lab, Toronto, Ontario, Canada

When a deepfake detector correctly flags a fake, does it also cite the *right* evidence?

Prediction accuracy can flatter a detector, because a model can be right about
*whether* a clip is fake while being wrong about *why*. This is a small, self-contained
probe of that gap: 200 FakeAVCeleb clips, an audio-visual VLM asked for a verdict plus
evidence drawn from a fixed taxonomy, and a scoring rule that needs no human
adjudication.

The trick is that **the way a clip was built determines which evidence could physically
be there.** A clip whose face is untouched cannot exhibit skin-boundary artifacts; a
clip whose audio is original cannot exhibit a lip-sync mismatch. So the quadrant labels
give free ground truth for the model's *reasoning*, not just its verdict.

| Quadrant | Construction | Admissible evidence |
|---|---|---|
| RVRA | unmodified | none — any citation is invented |
| FVRA | fsgan / wav2lip / faceswap | spatial-geometric, temporal, physiological |
| RVFA | rtvc voice clone | cross-modal only |
| FVFA | wav2lip + rtvc | cross-modal, spatial-geometric, temporal |

## Result

Qwen3-Omni-30B-A3B, 200 clips, zero-shot, greedy. Full numbers in
[`results/RESULTS.md`](results/RESULTS.md).

| Quadrant | Acc | Detected | Admissible evidence |
|---|---|---|---|
| RVRA | 0.98 | — | — |
| FVRA | 0.16 | 8/50 | **2/8** |
| RVFA | 0.24 | 12/50 | **12/12** |
| FVFA | 0.26 | 13/50 | **13/13** |
| All | 0.41 | 33/150 | 27/33 |

Detection is nearly flat across the three manipulated quadrants. Attribution is not: it
is perfect where the audio was manipulated and collapses where only the video was. On
those face-manipulated clips the model cited lip-sync mismatch **7 times out of 10 total
citations** — a signal that cannot be present, because the audio is unmodified and both
fsgan and wav2lip preserve mouth-audio agreement. The model defaults to the prototypical
deepfake explanation rather than the one it could have observed.

Ghost evidence on genuine clips is low (0.02), so the failure is not indiscriminate
hallucination. It is specifically a failure to pick the *right* explanation.

Qwen2.5-Omni-7B was also run and is **not a usable subject**: it returned `real` on
197/200 clips, detecting 3 of 150 fakes. A perception control (`probe.py`) confirms this
is a detection limit rather than an input failure — the same model describes the clips
and transcribes their speech accurately, and keeps the `real` verdict under a
forced-choice prompt that names the manipulation families.

## Run it

```bash
# 1. paths — edit the site-configuration block, or export over it
$EDITOR env.sh

# 2. environment (needs internet; ~5.5 GB venv)
bash setup.sh

# 3. weights, as a batch job (login nodes tend to kill a 23 GB pull)
sbatch slurm/download.sbatch

# 4. check everything before committing to a full run
sbatch slurm/preflight.sbatch

# 5. go. Submit from the repo root.
sbatch slurm/run.sbatch                 # Qwen2.5-Omni-7B
sbatch slurm/qwen3.sbatch               # Qwen3-Omni-30B-A3B, 2 GPUs
```

`preflight.py` checks packages, GPU, weights, dataset layout, and one end-to-end
inference, stopping at the first failure with the fix. Run it first; it costs minutes
and saves hours.

`run_vlm.py` is resumable — it appends to `preds.jsonl` and skips clips already there,
so a preempted or requeued job continues rather than restarting.

The SLURM headers carry placeholder `--account` / `--partition` values, flagged in each
script; set them for your scheduler or delete them.

## Files

| | |
|---|---|
| `taxonomy.py` | the evidence taxonomy and the quadrant → admissible-category map |
| `sample.py` | draw N clips per quadrant, spread across identities |
| `run_vlm.py` | one prompt per clip, append JSON lines |
| `score.py` | accuracy, attribution correctness/precision, ghost rate, bootstrap CIs |
| `by_method.py` | re-score against the dataset's own `meta_data.csv` labels |
| `probe.py` | perception control — does the model see and hear the clip at all |
| `preflight.py` | check every prerequisite before a run |
| `results/` | the outputs backing the table above, including every raw response |
| `paper/` | LaTeX for the results paragraph, tables, and protocol appendix |

## Notes on validity

**Label the clips from `meta_data.csv`, not from filenames.** `sample.py` infers the
manipulation from the filename, which misses `fsgan` entirely (those files carry no
method token) — 97 of our 200 clips were mislabelled that way. `by_method.py` re-scores
against the dataset's own labels; the result above survives it (fsgan 2/6, wav2lip 0/2).
The `method` column in `sample.csv` is therefore indicative only.

**Audio and video must share a time base.** `run_vlm.py` passes the *actual* per-clip
frame sampling rate to the processor. This is not cosmetic: the processor turns that
rate into `video_second_per_grid`, the grid TMRoPE interleaves audio against, and it
defaults to 2.0 if unset. Sampling at a different rate than the processor assumes lays
the frames over the wrong span, injecting an artificial audio-video desync into every
clip — which reads as exactly the `lip_sync_mismatch` signal this experiment turns on.

**One metric in `score.py` does not work.** "Accuracy by number of cited categories" was
intended as an evidence-sufficiency readout, but the prompt requires an empty evidence
list when the verdict is `real`, so "cited no category" and "returned real" are the same
event. The statistic reads 0.00 in the zero-category stratum and 1.00 elsewhere for any
model whatsoever. Testing sufficiency needs evidence elicited independently of the
verdict. The output is left in place, labelled, rather than silently dropped.

**`physiological` was never cited** — zero times across 400 responses, both models. At a
16-frame budget we cannot separate "not observable" from "not represented by the model".
A quarter of the taxonomy is untested.

## Caveats

Two models, one prompt, zero-shot, greedy decoding. FakeAVCeleb's generators date to
2021, so absolute detection rates are not the claim — the accuracy/attribution gap is.
Attribution is scored only over correctly detected fakes, so the FVRA cell rests on 8
clips and its interval is wide; raising `--per-quadrant` is the obvious first
improvement.
