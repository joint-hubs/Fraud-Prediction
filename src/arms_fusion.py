"""Latent-space fusion arm (FOC-178 F4): ONE joint representation per row,
built by statelessly concatenating the three F4 modality blocks —
face (512) + text (384) + demographic (384) = 1280 dims.

Implements the feature side of the `latent-fusion` arm of the unified runner
(fraud_pipeline.py). The registered arm's model is the SAME fixed XGBClassifier
factory as xgb-client (fraud_pipeline.xgb_params), so the arm's hypothesis is
"does the fused multi-modal latent space add signal over its single-modal
constituents", held by keeping the model identical.

Fusion choice: CONCAT of L2-normalized modality blocks — not a fitted
projection, not alignment. WHY, precisely:

- Any FITTED projection (PCA whitening, CCA, Procrustes/orthogonal alignment,
  a learned fusion MLP) must be fit SOMEWHERE. The runner contract evaluates
  build_features on the FULL enriched frame BEFORE the chronological split —
  anything fit in there sees test-set structure, i.e. leaks. A fold-internal
  refit instead (fit the projection inside make_model per split) would fix
  the leakage but makes the arm's feature matrix split-dependent, breaking
  the cached label-free extraction contract every other F4 arm follows.
- Alignment methods (CCA/Procrustes) additionally need PAIRED modalities of
  the same entity across domains; our modalities describe the same customer
  from three angles with no cross-domain anchor set, and 100 customers is far
  too small to fit a credible shared space.
- Concat of unit-norm blocks IS the leak-free "one latent space" baseline:
  every row lives in R^1280 with each modality contributing exactly unit
  length, so no modality dominates by scale, and cross-modal geometry
  (e.g. a face-text angle) is directly readable. Downstream here is a
  threshold-splitting XGB, which is per-feature monotone-invariant — block
  scaling would not change its trees anyway; the L2 normalization matters for
  GEOMETRY consumers (distance/angle thresholds), which is exactly what the
  F5 threshold layer is planned to need.

The three blocks are REUSED, never re-encoded: arms_face (hash-seeded
StyleGAN2 -> FaceNet, committed npz), arms_text (synthesized transaction text,
minilm-l6 — the nb14 winner), arms_demo (dim_customer profile text, the same
minilm-l6 backbone). All three append_features are label-free and
fold-independent, so the fusion is too: supports_cv=True.

Column names: the modality prefixes ARE the block identity
(face_emb_000..511, txt_emb_000..383, demo_emb_000..383) — no renaming, so a
consumer can attribute any fused dimension to its modality by prefix.

Dependencies are optional on other machines: nothing heavy imports at module
import time. check_dependencies() is the AND of the three constituent probes
(first failure returned) — the runner's factory turns a non-None result into
a SKIPPED row (fraud_pipeline.ArmSkipped), never a crash.

Notebook API:
  from arms_fusion import append_features, FUSED_FEATURES
  enriched_fused = append_features(enriched)          # all three caches
"""

import pandas as pd

import arms_demo  # module level is light (numpy/pandas/hashlib only)
import arms_face
import arms_text

RANDOM_SEED = 42

# The fusion consumes exactly the nb14-chosen text checkpoint — frozen here so
# the arm name/description and the features cannot drift apart.
FUSION_TEXT_MODEL_TAG = arms_text.DEFAULT_MODEL  # 'minilm-l6'

FACE_FEATURES = list(arms_face.FACE_FEATURES)
TEXT_FEATURES = list(arms_text.text_feature_columns())
DEMO_FEATURES = list(arms_demo.demo_feature_columns())
FUSED_FEATURES = FACE_FEATURES + TEXT_FEATURES + DEMO_FEATURES


def check_dependencies():
    # AND of the three modality probes — the first failure message wins (any
    # one missing constituent makes the fused space unbuildable).
    for label, probe in (
        ("face", arms_face.check_dependencies),
        ("text", arms_text.check_dependencies),
        ("demo", arms_demo.check_dependencies),
    ):
        missing = probe()
        if missing is not None:
            return "%s: %s" % (label, missing)
    return None


def append_features(frame, use_cache=True):
    """frame + the three modality blocks concatenated = the fused latent space.

    Each constituent append_features returns a NEW frame and is itself cached
    (face npz + process cache; text corpus fingerprint; demo corpus digest),
    so a fusion build costs three cache reads after the first pass.
    use_cache=False forces fresh constituent passes (the determinism lever).
    Returns a NEW frame; the input is never mutated.
    """
    with_face = arms_face.append_features(frame, use_cache=use_cache)
    with_text = arms_text.append_features(
        with_face, model_name=FUSION_TEXT_MODEL_TAG, use_cache=use_cache
    )
    with_demo = arms_demo.append_features(with_text, use_cache=use_cache)
    present = [c for c in FUSED_FEATURES if c in with_demo.columns]
    assert len(present) == len(FUSED_FEATURES), (
        "fused space incomplete: %d of %d modality columns present"
        % (len(present), len(FUSED_FEATURES))
    )
    return with_demo
