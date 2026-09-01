"""Text-embedding feature arm (FOC-178 F4): sentence-transformers MiniLM as a
feature extractor, never a classifier.

Implements the feature side of the `text-features` arm family of the unified
runner (fraud_pipeline.py). The registered arm's model is the SAME fixed
XGBClassifier factory as xgb-client (fraud_pipeline.xgb_params), so the arm's
hypothesis is "do transaction-text embeddings add signal", held by keeping the
model identical. The encoder itself is only a feature extractor.

Synthetic-text caveat (decision D4). data/all_trxns.csv carries NO free-text
column — text is EXPERIMENTAL here, so every transaction's text is SYNTHESIZED
deterministically from fields the table already has (type, ccy, amount_eur,
counterparty, counterparty_country, customer_country; the customer id and the
timestamp seed the variation). The embeddings therefore largely RE-ENCODE
categorical/numeric signal the tabular arms already see, and any result speaks
to METHOD PLUMBING (does the extract-embed-append pipeline work end to end),
not to the real-world value of text on fraud. A genuine text arm needs a real
free-text field (dispute notes, merchant descriptors) this table does not have.

Synthesis scheme (pure function of the row's content):

- seed: sha256("<customer>|<timestamp isoformat>") -> 64-bit int. WHY hashlib
  and not built-in hash(): str hashing is salted per process, so a hash-seeded
  text would differ between runs. A content-derived seed makes the text a pure
  function of the row — stable across re-sorts, re-loads and shuffles, and
  never dependent on enumeration order.
- variation: one of 3 sentence templates (a narrative, a sender-perspective and
  a pipe-delimited variant), picked by a random.Random(seed) draw;
- slots: a per-type subject phrase (8 types), the amount as "<CCY> 34,814.29",
  both account ids with their countries, and a size descriptor banded on
  amount_eur with FIXED thresholds (<1k low-value, <10k moderate-value,
  <100k high-value, else very-high-value).
- no label anywhere: fraud_flag is not an input to the synthesis.

Encoding: the two cached sentence-transformers MiniLM checkpoints
(all-MiniLM-L6-v2 / all-MiniLM-L12-v2, both 384-dim) are the ONLY candidates —
bigger models would have to download at run time, and this machine's link plus
the 600 s cell ceiling forbid any network fetch inside an executed cell. Both
snapshots are probed with local_files_only=True and loaded from the resolved
local snapshot path, so NO network call can happen at run time (the same
checkpoint discipline as arms_timesfm). Embeddings are L2-normalized
(normalize_embeddings=True): every row is a unit vector, so tree splits read
cosine geometry directly.

Determinism: fixed seed, eval-mode inference (st encode runs the model in eval
mode, no dropout), fixed batch composition for a fixed input order. Two encode
passes over the SAME row order are bit-identical; a DIFFERENT row order changes
batch composition and may shift floats at epsilon scale — nb14's determinism
check therefore encodes a SHUFFLED copy and aligns the comparison on row
IDENTITY (frame index), never positionally (the F3 round-2 lesson: a positional
diff on differently-ordered frames reported a bogus 8.1e+01).

Caching: one encode pass per process, cached on a fingerprint of (model, text
content, frame index) — the same precedent as the timesfm feature cache. The
extraction is label-free and fold-independent (a row's text and embedding never
depend on which split/fold it lands in), so CV refits only the XGB:
supports_cv=True.

Column names: txt_emb_000..txt_emb_383 — XGBoost-safe (no '[', ']', '<').

st 6.x API note: get_sentence_embedding_dimension() was renamed
get_embedding_dimension(); this module never calls either — the dimension is
read off the encoded matrix shape, which is version-proof.

Dependencies are optional on other machines: nothing imports
sentence_transformers/torch at module import time. check_dependencies() probes
them plus both model caches and returns a short message instead of raising —
the runner's factory turns a non-None result into a SKIPPED row
(fraud_pipeline.ArmSkipped), never a crash.

Notebook API:
  from arms_text import append_features, encode_texts, synthesize_texts
  enriched_txt = append_features(enriched, model_name="minilm-l6")   # cached
  matrix = encode_texts(synthesize_texts(enriched), "minilm-l6")     # cache bypass
"""

import hashlib
import random

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# The ONLY candidates (both pre-cached in the local HF cache — see the module
# docstring for the no-download constraint). Keys are the model tags used in
# arm names: text-features-<tag>.
CACHED_MODELS = {
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
}
DEFAULT_MODEL = "minilm-l6"

# Batch size for one encode pass: 5302 short sentences in 21 GPU batches —
# seconds on CUDA, and a FIXED batch composition for a fixed input order.
TEXT_ENCODE_BATCH_SIZE = 256

# Columns the synthesis reads (the text must depend on nothing else; the cache
# fingerprint below covers exactly the synthesized text, so this list documents
# the dependency rather than being load-bearing).
TEXT_SOURCE_COLUMNS = [
    "customer",
    "timestamp",
    "type",
    "ccy",
    "amount_eur",
    "counterparty",
    "counterparty_country",
    "customer_country",
]

# Feature-column prefix; names are XGBoost-safe by construction (no '['/']'/'<').
TEXT_FEATURE_PREFIX = "txt_emb_"

# Documented MiniLM output width — asserted against the encoded matrix in
# append_features, never trusted blindly (the dimension is read off the matrix).
EMBEDDING_DIM = 384

# Per-type subject phrase: the sentence's head noun. A transaction's `type`
# already says what kind of movement it is — the text makes it SAY so.
_TYPE_SUBJECT = {
    "BILLING": "billing charge",
    "DIVIDEND": "dividend payout",
    "INTEREST": "interest payment",
    "INVESTMENT": "investment transaction",
    "OTHER": "miscellaneous transaction",
    "PAYMENT": "payment",
    "TRANSFER": "account transfer",
    "TT": "telegraphic transfer",
}

# Three sentence shapes per row; the seeded draw picks one. Same slots in all
# three, different surface forms — enough variation for the encoder to see
# sentence structure, not enough to smuggle in any new information.
_TEXT_TEMPLATES = (
    "{subject_cap} of {amount_txt} from customer {customer} ({customer_country}) "
    "to counterparty {counterparty} ({counterparty_country}); {size_txt} transaction.",
    "Customer {customer} in {customer_country} sent {amount_txt} as a {subject} "
    "to counterparty {counterparty} based in {counterparty_country} ({size_txt}).",
    "{subject_cap} | {customer} ({customer_country}) -> {counterparty} "
    "({counterparty_country}) | {amount_txt} | {size_txt}",
)

# Size descriptor bands on amount_eur — FIXED thresholds (not data-derived
# quantiles) so the wording is a stable function of the amount alone.
_SIZE_BANDS = (
    (1000.0, "low-value"),
    (10000.0, "moderate-value"),
    (100000.0, "high-value"),
)

_ENCODER_CACHE = {}
_EMBED_CACHE = {}


def _content_seed(customer, timestamp):
    # 64-bit seed from the row's CONTENT (customer id + timestamp). WHY sha256
    # and not built-in hash(): str hashing is salted per process (PYTHONHASHSEED),
    # so a hash-derived seed would make the text differ between runs. Content
    # only — the row's enumeration order never enters the seed.
    payload = "%s|%s" % (customer, pd.Timestamp(timestamp).isoformat())
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _size_text(amount_eur):
    for bound, label in _SIZE_BANDS:
        if amount_eur < bound:
            return label
    return "very-high-value"


def synthesize_transaction_text(
    customer, timestamp, txn_type, ccy, amount_eur, counterparty,
    counterparty_country, customer_country,
):
    """One transaction's synthesized description — a pure function of its content.

    Deterministic: the seeded draw (template choice) is keyed on
    sha256(customer|timestamp); every other slot is the row's own values.
    """
    assert txn_type in _TYPE_SUBJECT, "unknown transaction type: %r" % txn_type
    rng = random.Random(_content_seed(customer, timestamp))
    template = _TEXT_TEMPLATES[rng.randrange(len(_TEXT_TEMPLATES))]
    return template.format(
        subject=_TYPE_SUBJECT[txn_type],
        subject_cap=_TYPE_SUBJECT[txn_type].capitalize(),
        amount_txt="%s %s" % (ccy, format(float(amount_eur), ",.2f")),
        customer=str(customer),
        customer_country=str(customer_country),
        counterparty=str(counterparty),
        counterparty_country=str(counterparty_country),
        size_txt=_size_text(float(amount_eur)),
    )


def synthesize_texts(frame):
    """The synthesized description for every row of `frame`, in frame row order.

    `frame` needs TEXT_SOURCE_COLUMNS. Asserts the data invariants up front —
    a missing column or a NaN amount is a caller bug, not an embedding concern.
    """
    missing = [c for c in TEXT_SOURCE_COLUMNS if c not in frame.columns]
    assert not missing, "text synthesis missing columns: %s" % missing
    assert frame["amount_eur"].notna().all(), "amount_eur has NaN — cannot describe the amount"
    return [
        synthesize_transaction_text(
            row.customer, row.timestamp, row.type, row.ccy, row.amount_eur,
            row.counterparty, row.counterparty_country, row.customer_country,
        )
        for row in frame[TEXT_SOURCE_COLUMNS].itertuples(index=False)
    ]


def text_feature_columns(n_dims=EMBEDDING_DIM):
    # XGBoost-safe embedding column names (no '[', ']', '<' — encode_dummies'
    # sanitization rule; txt_emb_%03d never needs it).
    return ["%s%03d" % (TEXT_FEATURE_PREFIX, i) for i in range(n_dims)]


def check_dependencies():
    # Probe-import the heavy libraries, then probe BOTH model caches WITHOUT
    # network (local_files_only snapshot_download raises when a snapshot is
    # absent). Returns None when everything is in place, otherwise a short
    # message — the runner's factory turns a non-None result into an ArmSkipped
    # row (both in build_features and make_model; build_features runs first).
    missing = []
    for module in ("sentence_transformers", "torch", "huggingface_hub"):
        try:
            __import__(module)
        except ImportError as exc:
            missing.append("%s (%s)" % (module, exc))
    if missing:
        return "; ".join(missing)
    from huggingface_hub import snapshot_download  # lazy: optional dependency

    for tag, repo_id in sorted(CACHED_MODELS.items()):
        try:
            snapshot_download(repo_id, local_files_only=True)
        except Exception as exc:
            return "model %s (%s) not in the local HF cache (%s: %s)" % (
                tag, repo_id, type(exc).__name__, exc,
            )
    return None


def get_encoder(model_name=DEFAULT_MODEL):
    # Load one cached checkpoint once per process (registry path: the notebook's
    # text arm -> build_features -> append_features). The snapshot is resolved
    # with local_files_only FIRST and the LOCAL PATH is handed to
    # SentenceTransformer, so no code path can reach the network.
    if model_name not in CACHED_MODELS:
        raise ValueError(
            "unknown text model: %s (cached candidates: %s)"
            % (model_name, sorted(CACHED_MODELS))
        )
    if model_name not in _ENCODER_CACHE:
        import torch  # lazy: optional dependency
        from huggingface_hub import snapshot_download  # lazy
        from sentence_transformers import SentenceTransformer  # lazy

        torch.manual_seed(RANDOM_SEED)
        snapshot = snapshot_download(CACHED_MODELS[model_name], local_files_only=True)
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        _ENCODER_CACHE[model_name] = SentenceTransformer(snapshot, device=device_name)
    return _ENCODER_CACHE[model_name]


def encode_texts(texts, model_name=DEFAULT_MODEL):
    """L2-normalized embeddings for `texts`, one batched GPU pass.

    Deterministic for a fixed input order (eval mode, fixed batch size); a
    different input order may shift floats at epsilon scale — nb14's
    determinism check aligns by row identity and reports the observed delta.
    Returns a float32 (n_texts, dim) ndarray.
    """
    model = get_encoder(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=TEXT_ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    assert matrix.shape[0] == len(texts), "encode lost rows"
    return matrix


def append_features(frame, model_name=DEFAULT_MODEL, use_cache=True):
    """frame + the model's embedding columns, encoded once and cached.

    The cache key is a fingerprint of (model, synthesized text content, frame
    index) — the exact inputs the features depend on — so any re-load of the
    same table reuses the GPU pass while a re-indexed or edited table re-encodes
    instead of silently returning misaligned rows. use_cache=False forces a
    fresh pass (nb14's determinism check). Returns a NEW frame — the input is
    never mutated.
    """
    texts = synthesize_texts(frame)
    key = None
    if use_cache:
        assert frame.index.is_unique, "frame row index not unique — cannot align"
        text_digest = hashlib.sha256("\x1e".join(texts).encode("utf-8")).hexdigest()
        index_digest = hashlib.sha256(
            np.asarray(frame.index).tobytes()
        ).hexdigest()
        key = (model_name, len(texts), text_digest, index_digest)
        cached = _EMBED_CACHE.get(key)
        if cached is not None:
            return pd.concat([frame, cached], axis=1)
    matrix = encode_texts(texts, model_name)
    assert matrix.shape[1] == EMBEDDING_DIM, (
        "unexpected embedding width %d (expected %d)" % (matrix.shape[1], EMBEDDING_DIM)
    )
    features = pd.DataFrame(
        matrix, index=frame.index, columns=text_feature_columns(matrix.shape[1])
    )
    if use_cache:
        _EMBED_CACHE[key] = features
    return pd.concat([frame, features], axis=1)
