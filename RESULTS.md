# Results: attribution correctness on FakeAVCeleb

Run 2026-09-04 on a SLURM cluster, 1-2x A100-80GB. Two audio-visual VLMs, 200 clips, 50 per quadrant,
zero-shot, one prompt, greedy decoding.

## Headline

| Model | Acc | Recall (fakes) | Attr. correct | Attr. precision | Ghost evidence |
|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | 0.27 | 0.02 | 1.00 (n=3) | 0.80 | 0.00 |
| Qwen3-Omni-30B-A3B | 0.41 [0.34, 0.48] | 0.22 | 0.82 [0.68, 0.94] | 0.82 [0.69, 0.93] | 0.02 [0.00, 0.07] |

95% CIs are 1000-sample bootstrap over clips.

**Qwen2.5-Omni-7B is not a usable subject.** It answered `real` on 197/200 clips,
including 147/150 fakes. Its attribution numbers are computed over the 3 fakes it
caught and should not be quoted. This is a genuine model limit, not a harness failure:
on a perception control the same model describes the clips accurately ("a man with
short dark hair, wearing a black hoodie") and transcribes the speech verbatim, and
under a forced-choice prompt naming the manipulations it still answers "Real ... no
obvious signs of manipulation like unnatural lip sync or cloned voice" on wav2lip
fakes. It sees and hears the clip and is simply wrong.

Everything below is Qwen3-Omni-30B-A3B.

## The gap

| Quadrant | Manipulation | n | Acc | Attr. correct | Attr. precision |
|---|---|---|---|---|---|
| RVRA | none | 50 | 0.98 | -- | -- |
| FVRA | fsgan / wav2lip / faceswap | 50 | 0.16 | **0.25** | 0.30 |
| RVFA | rtvc voice clone | 50 | 0.24 | **1.00** | 1.00 |
| FVFA | wav2lip + rtvc | 50 | 0.26 | **1.00** | 1.00 |

Detection is roughly flat across the three fake quadrants (0.16-0.26). Attribution is
not: it is perfect where the audio was touched and collapses to 0.25 where only the
video was.

The mechanism is visible in the citation counts (`*` = admissible for that quadrant):

```
FVRA: spatial_geometric*=2, temporal*=1, physiological*=0, cross_modal=7
RVFA: spatial_geometric=0,  temporal=0,  physiological=0,  cross_modal*=12
FVFA: spatial_geometric*=2, temporal*=2, physiological=0,  cross_modal*=13
RVRA: spatial_geometric=0,  temporal=0,  physiological=0,  cross_modal=1
```

On FVRA the model cited `cross_modal` (lip-sync mismatch) **7 times out of 10 total
citations**. FVRA clips carry their original untouched audio, and both fsgan and
wav2lip preserve mouth-audio agreement, so a lip-sync mismatch cannot be present. The
model reached for the most familiar deepfake explanation rather than the one it could
have observed.

This survives correction of the dataset labels. `sample.py` infers the method from the
filename, which misses fsgan entirely; re-scored against `meta_data.csv` (see
`by_method.py`), the FVRA result splits as:

| True method | n | detected | right reason | citations |
|---|---|---|---|---|
| fsgan | 21 | 6 | 2/6 | spatial=2, temporal=1, **cross_modal=5** |
| wav2lip | 27 | 2 | 0/2 | **cross_modal=2** |
| faceswap | 2 | 0 | -- | -- |

Both families leave the audio untouched, so the phantom citation holds under either
labelling.

## Evidence sufficiency

Recall conditioned on how many categories were cited, over true fakes:

```
0 categories: n=117, recall=0.00
1 category:   n=29,  recall=1.00
2 categories: n=2,   recall=1.00
3 categories: n=2,   recall=1.00
```

This is definitional, not empirical -- the prompt requires an empty evidence list when
the verdict is `real`, so "cited nothing" and "said real" are the same event. The row
is not evidence that more citations imply better detection, and should not be read as
a sufficiency result.

## Ghost evidence

0.02 on reals: one RVRA clip out of 50 drew a `cross_modal` citation. Hallucinated
attribution on genuine media is rare for this model. The interesting failure is not
inventing evidence where none exists, but citing the *wrong kind* where some does.

## Caveats

- Two models, one prompt, zero-shot, greedy. Absolute detection rates are not the claim.
- FakeAVCeleb generators are 2021-era. The acc/attribution gap is the claim.
- `physiological` drew **zero citations in 400 answers**. A quarter of the taxonomy went
  unused, so nothing here speaks to whether those signals are observable or scoreable.
  At 16 frames / 2 fps they may simply not be.
- Attribution is measured only over correctly-detected fakes, so n is small: 33 for
  Qwen3-Omni (8 of them FVRA). The CIs above reflect that.
- `EXPECTED[FVRA]` admits `physiological`, which wav2lip cannot produce (it edits the
  mouth only). This does not affect any number reported here, because `physiological`
  was never cited, but the mapping is coarse and should be stated as such.
