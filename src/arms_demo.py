"""Demographic-embedding feature arm (FOC-178 F4): each customer's
dim_customer profile rendered as a short deterministic text and embedded with
the SAME MiniLM-L6 backbone the text arm uses — one text encoder across the
modalities, so modality differences come from the CONTENT, not the encoder.

Implements the feature side of the `demo-features` arm of the unified runner
(fraud_pipeline.py). The registered arm's model is the SAME fixed XGBClassifier
factory as xgb-client (fraud_pipeline.xgb_params), so the arm's hypothesis is
"does a dense demographic-profile embedding add signal over the raw one-hot
demographics the client arm already has", held by keeping the model identical.
The encoder is only a feature extractor.

Why an embedding of 8 categorical fields at all. xgb-client already one-hot
encodes exactly these fields (CLIENT_CATEGORICAL), so the honest hypothesis is
NOT "new information" but "new GEOMETRY": the encoder maps every profile to a
dense unit vector whose neighbourhood structure (similar profiles close
together) a tree ensemble cannot get from orthogonal one-hots. At 100
customers the embedding is a re-encoding of 8 fields the tabular arm already
sees — pre-registered expectation: no gain. The arm exists to complete the
latent space for fusion (nb15), where the demographic block is a modality on
equal footing with face and text, and to keep the method plumbing comparable
for when dim_customer grows real fields.

Profile text (pure function of the profile, no RNG anywhere):

  "Customer profile: gender M, age 39, unemployed in the technology industry,
   income band 2_20k_40k, account tenure 9 years, usually on mobile via branch."

The 8 source fields are the dim_customer columns the runner's load_enriched
already merged onto the frame (gender, age, employment_industry,
employment_status, account_tenure_years, income_band, device, channel). They
are CONSTANT per customer (many_to_one validated join), asserted here — the
profile text is rendered from the customer's first row, so a violated
constancy would silently pick an arbitrary row. Raw values, no paraphrase:
the labels are already short, and any paraphrase would be OUR invention, not
data. No label anywhere: fraud_flag is never read, and fold membership cannot
change a value — the embedding is a pure function of the customer id's
profile, which is why the arm ships supports_cv=True.

Encoding: arms_text.encode_texts on the minilm-l6 checkpoint (the nb14 winner
— chosen on the PRIMARY axis within the noise budget over the stronger-but-
wider-CI l12, so the cheapest cached encoder carries all F4 text-shaped
modalities). 100 short profiles = one batch, seconds on CUDA. L2-normalized,
same as every other F4 block.

Caching: one encode pass per process keyed on a digest of the rendered
profile corpus (content-addressed — an edited dim_customer re-encodes instead
of silently returning stale vectors), plus a committed artifact
data/demo_embeddings.npz (customer_id + N x 384 float32 + the profile scheme
string + the corpus digest), so nb15/the runner load without a GPU pass. The
npz is trusted only when its digest matches the current corpus.

Column names: demo_emb_000..demo_emb_383 — XGBoost-safe (no '[', ']', '<').

Dependencies are optional on other machines: nothing imports
sentence_transformers/torch at module import time. check_dependencies()
delegates to arms_text's probe (same encoder, same cache discipline) and
returns a short message instead of raising — the runner's factory turns a
non-None result into a SKIPPED row (fraud_pipeline.ArmSkipped), never a crash.

Notebook API:
  from arms_demo import append_features, embed_customers
  enriched_demo = append_features(enriched)                  # cached
  vecs = embed_customers(["C123...", ...], use_cache=False)  # cache bypass
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# The demographic block rides the SAME encoder checkpoint as the chosen text
# arm (nb14: text-features-minilm-l6). One text backbone across modalities.
DEMO_MODEL_TAG = "minilm-l6"

# Columns the profile text renders (all merged onto the frame by
# load_enriched from dim_customer; constant per customer).
PROFILE_COLUMNS = [
    "gender",
    "age",
    "employment_industry",
    "employment_status",
    "account_tenure_years",
    "income_band",
    "device",
    "channel",
]

# Profile-text template — one fixed shape, no seeded variation. Variation
# would add encoder-visible surface diversity but zero new information (the
# slots ARE the information); the plain template keeps the block a pure
# function of the 8 fields.
_PROFILE_TEMPLATE = (
    "Customer profile: gender {gender}, age {age}, {employment_status} in the "
    "{employment_industry} industry, income band {income_band}, account tenure "
    "{account_tenure_years} years, usually on {device}, {channel} channel."
)

# Feature-column prefix; names are XGBoost-safe by construction.
DEMO_FEATURE_PREFIX = "demo_emb_"

# Documented MiniLM output width — asserted against the encoded matrix in
# append_features, never trusted blindly (the dimension is read off the matrix).
EMBEDDING_DIM = 384

# Committed artifact (same shape conventions as the face npz).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEMO_EMBEDDINGS_NPZ_PATH = DATA_DIR / "demo_embeddings.npz"

_EMBEDDING_CACHE = {}  # corpus digest -> customer-indexed feature frame


def profile_text(gender, age, employment_industry, employment_status,
                 account_tenure_years, income_band, device, channel):
    """One customer's rendered profile — a pure function of the 8 fields."""
    return _PROFILE_TEMPLATE.format(
        gender=gender,
        age=age,
        employment_industry=employment_industry,
        employment_status=employment_status,
        account_tenure_years=account_tenure_years,
        income_band=income_band,
        device=device,
        channel=channel,
    )


def customer_profiles(frame):
    """Per-customer profile texts from `frame`, in sorted customer order.

    Asserts the dim_customer join invariants up front: every profile column
    present, no NaNs, and constant within each customer (the frame is a
    many_to_one merge — a violated constancy would make the rendered profile
    depend on which row happened to be first).
    """
    missing = [c for c in PROFILE_COLUMNS if c not in frame.columns]
    assert not missing, "demographic profile missing columns: %s" % missing
    per_customer = frame.groupby("customer")[PROFILE_COLUMNS].nunique(dropna=False)
    bad = per_customer[(per_customer > 1).any(axis=1)]
    assert bad.empty, "profile fields not constant per customer: %s" % list(bad.index[:5])
    first = frame.groupby("customer")[PROFILE_COLUMNS].first().sort_index()
    assert first.notna().all().all(), "NaN in demographic profile fields"
    texts = [
        profile_text(**row._asdict())
        for row in first[PROFILE_COLUMNS].itertuples(index=False, name="profile")
    ]
    return list(first.index), texts


def demo_feature_columns(n_dims=EMBEDDING_DIM):
    # XGBoost-safe embedding column names (no '[', ']', '<').
    return ["%s%03d" % (DEMO_FEATURE_PREFIX, i) for i in range(n_dims)]


def _corpus_digest(customer_ids, texts):
    # Content-addressed cache key: the rendered corpus, order-independent
    # (sorted ids first) so a re-loaded frame with the same dim_customer hits.
    payload = "\x1e".join("%s\x1f%s" % (i, t) for i, t in zip(customer_ids, texts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_npz(path):
    # Shared npz reader: returns (ids, embeddings float32, scheme, digest)
    # or None when the artifact does not exist yet.
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return (
            list(data["customer_id"]),
            data["embeddings"].astype(np.float32),
            str(data["profile_scheme"][0]),
            str(data["corpus_digest"][0]),
        )


def save_embeddings(feature_frame, corpus_digest, path=DEMO_EMBEDDINGS_NPZ_PATH):
    # Persist the committed embedding artifact: customer_id + N x 384 float32
    # + the profile scheme and corpus digest, so provenance rides inside the
    # file and a stale npz is detected instead of silently consumed.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        customer_id=np.asarray(feature_frame.index, dtype="U"),
        embeddings=feature_frame.to_numpy(dtype=np.float32),
        profile_scheme=np.asarray([_PROFILE_TEMPLATE], dtype="U"),
        corpus_digest=np.asarray([corpus_digest], dtype="U"),
    )


def embed_customers(customer_ids, texts, corpus_digest, use_cache=True,
                    npz_path=DEMO_EMBEDDINGS_NPZ_PATH):
    """customer-indexed feature frame for the requested ids (encode or load).

    Resolution order: process cache (corpus digest) -> committed npz (digest
    must match the current corpus) -> fresh arms_text encode pass. Returns a
    DataFrame indexed by customer_id with DEMO EMBEDDING columns.
    """
    import arms_text  # lazy sibling: heavy imports (torch/st) live there

    customer_ids = sorted(set(str(c) for c in customer_ids))
    if not customer_ids:
        return pd.DataFrame(columns=demo_feature_columns())
    if use_cache and corpus_digest in _EMBEDDING_CACHE:
        cached = _EMBEDDING_CACHE[corpus_digest]
        if set(customer_ids).issubset(cached.index):
            return cached.loc[customer_ids]
    read = _read_npz(npz_path) if use_cache else None
    if read is not None:
        ids, embeddings, _scheme, npz_digest = read
        if npz_digest == corpus_digest:
            frame = pd.DataFrame(embeddings, index=ids,
                                 columns=demo_feature_columns(embeddings.shape[1]))
            if set(customer_ids).issubset(frame.index):
                if corpus_digest not in _EMBEDDING_CACHE:
                    _EMBEDDING_CACHE[corpus_digest] = frame
                return frame.loc[customer_ids]
    # Fresh pass: render order matches customer_ids (sorted), encode once.
    id_to_text = dict(zip(customer_ids, texts))
    ordered_texts = [id_to_text[c] for c in customer_ids]
    matrix = arms_text.encode_texts(ordered_texts, DEMO_MODEL_TAG)
    assert matrix.shape == (len(customer_ids), EMBEDDING_DIM), (
        "unexpected demo embedding shape %s (expected %s)"
        % (matrix.shape, (len(customer_ids), EMBEDDING_DIM))
    )
    frame = pd.DataFrame(
        matrix, index=customer_ids, columns=demo_feature_columns(matrix.shape[1])
    )
    if use_cache:
        _EMBEDDING_CACHE[corpus_digest] = frame
        save_embeddings(frame, npz_path)
    return frame


def check_dependencies():
    # The demographic block encodes through arms_text's checkpoint — its probe
    # already covers sentence_transformers/torch/huggingface_hub AND the model
    # cache discipline (local_files_only). Nothing else is needed.
    import arms_text  # lazy sibling

    return arms_text.check_dependencies()


def append_features(frame, use_cache=True, npz_path=DEMO_EMBEDDINGS_NPZ_PATH):
    """frame + the DEMO_FEATURES columns, one embedding broadcast per customer.

    The embedding is a function of the customer's dim_customer profile ALONE
    (rendered text -> MiniLM), so broadcasting is a plain per-customer constant
    join — every row of a customer carries that customer's 384-d unit vector.
    Zero label access, fold-independent: supports_cv=True. Returns a NEW
    frame; the input is never mutated. use_cache=False bypasses both the
    process cache and the npz (forces a fresh encode — the determinism lever).
    """
    customer_ids, texts = customer_profiles(frame)
    digest = _corpus_digest(customer_ids, texts)
    embeddings = embed_customers(customer_ids, texts, digest,
                                 use_cache=use_cache, npz_path=npz_path)
    assert set(customer_ids).issubset(embeddings.index), (
        "embedding table missing customers"
    )
    features = embeddings.loc[frame["customer"].astype(str).to_numpy()]
    features.index = frame.index
    return pd.concat([frame, features], axis=1)
