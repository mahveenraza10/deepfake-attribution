"""Content-level evidence taxonomy (Table A1) and ground-truth expectations per FakeAVCeleb quadrant."""
import os


def clip_path(p, root=None):
    """Resolve a clip reference to a readable path.

    Manifests store paths relative to the FakeAVCeleb root so that a sample or a
    predictions file stays valid on someone else's filesystem. Absolute paths are
    passed through, so an external manifest still works.
    """
    if os.path.isabs(p):
        return p
    return os.path.join(root or os.environ.get("FAVC", ""), p)


CATEGORIES = ["spatial_geometric", "temporal", "physiological", "cross_modal"]

SIGNALS = {
    "spatial_geometric": [
        "boundary_artifacts", "dermal_uniformity", "specular_disparity",
        "semantic_inconsistency", "facial_symmetry_drift", "gaze_implausibility",
    ],
    "temporal": [
        "blink_frequency", "colour_flickering", "motion_deformation", "identity_jitter",
    ],
    "physiological": [
        "rppg_disruption", "pupillary_response", "micro_expression_absence",
    ],
    "cross_modal": [
        "lip_sync_mismatch", "phoneme_viseme_inconsistency", "inconsistent_respiration",
    ],
}

# Which evidence categories are *legitimately* observable given how each quadrant was made.
# FakeAVCeleb: FVRA = faceswap / fsgan (visual only); RVFA = RTVC voice clone (audio only,
# so the only visible trace is audio-visual desynchrony); FVFA = wav2lip + RTVC.
EXPECTED = {
    "RVRA": set(),
    "FVRA": {"spatial_geometric", "temporal", "physiological"},
    "RVFA": {"cross_modal"},
    "FVFA": {"cross_modal", "spatial_geometric", "temporal"},
}

QUADRANT_DIRS = {
    "RealVideo-RealAudio": "RVRA",
    "FakeVideo-RealAudio": "FVRA",
    "RealVideo-FakeAudio": "RVFA",
    "FakeVideo-FakeAudio": "FVFA",
}

PROMPT = """You are a forensic analyst verifying whether a short talking-head video is authentic or synthetic.
Watch the video and listen to the audio. Then answer in JSON only, with this schema:

{{
  "verdict": "real" | "fake",
  "confidence": 0.0-1.0,
  "evidence": [
    {{"category": <one of {cats}>, "signal": <one of the signal names below>, "note": <one sentence, what you observed and where>}}
  ]
}}

Rules:
- If verdict is "real", evidence must be an empty list.
- Cite only evidence you actually observed. Do not list a signal because it is typical of deepfakes.
- Categories and signals:
{signals}
"""


def build_prompt():
    sig_lines = "\n".join(f"  {c}: {', '.join(SIGNALS[c])}" for c in CATEGORIES)
    return PROMPT.format(cats=" | ".join(f'"{c}"' for c in CATEGORIES), signals=sig_lines)
