"""Face-embedding feature arm (FOC-178 F4): a face, not a forecast.

Implements the `face-features` arm of the unified runner (fraud_pipeline.py).
Each of the 100 customers receives ONE synthetic face: a StyleGAN2 (FFHQ
weights, locally cached pkl) image whose latent vector is derived
deterministically from the customer id — no network fetch, no real person.
A pretrained FaceNet (InceptionResnetV1, vggface2) embeds that image into a
512-d vector, and the customer's vector is broadcast onto every one of their
transaction rows as extra columns of the xgb-client matrix. The registered
arm's model is the SAME fixed XGBClassifier factory as xgb-client
(fraud_pipeline.xgb_params), so the arm's hypothesis — "does an arbitrary
per-customer face embedding add signal" — is held by keeping the model
identical (the timesfm-arm discipline, fraud_pipeline.py:540-584).

ETHICS (read before reusing any of this outside this repo): a face-to-fraud
score is ethically indefensible as a production tool — face appearance
correlates with protected attributes (age, gender, ethnicity proxies), so the
score would discriminate on appearance with no ex-ante individual
justification. Synthetic faces avoid harm to real people but NOT the
mechanism critique: the proxy logic itself is what would ship. This arm
exists ONLY as a methods-comparability experiment with a PRE-REGISTERED NULL
expectation (the F4 plan, section 7: appearance carries NO fraud signal), and
its honest scientific reading is that a per-customer face embedding is at
best a customer-identity proxy:

- on the two customer-grouped axes a test customer was never seen in training,
  so its face embedding is an arbitrary unseen constant — no transfer possible;
- on the chronological axis a test customer usually WAS seen in train, so the
  shared face is an identity-memorization channel — any lift there would be
  leakage by design, evidence about identity transfer, never about appearance.

Determinism (decision D5, exact rule): the latent seed for customer C is

  seed = int.from_bytes(sha256(("face-arm-v1:" + C).encode("utf-8")).digest()[:4], "big")

(the "face-arm-v1:" prefix domain-separates the hash from any other
customer-keyed seed in the repo; the FIRST 4 BYTES are read because numpy's
legacy RandomState accepts only 0..2**32-1); z =
numpy.random.RandomState(seed).randn(1, G.z_dim); w = G.mapping(z, None,
truncation_psi=0.7); img = G.synthesis(w, noise_mode="const",
force_fp32=True). No torch RNG is consumed (the const
noise buffers are baked into the pkl), so the same customer_id yields the
same pixels on this machine. DETERMINISM IS ANCHORED TO THE CACHE: the
committed JPEG under data/faces/ is the canonical image, and the committed
npz (data/face_embeddings.npz) is the canonical embedding — re-runs load
these artifacts, so byte-identity across runs holds by construction, and the
nb13 check verifies generate-then-embed reproduces the cache byte-for-byte
plus aligned-by-customer-id embedding equality (NEVER positional — the F3
review caught positional diffing reporting a bogus 8.1e+01 "GPU
nondeterminism" delta on misaligned rows).

Cache discipline:

- images: data/faces/<customer_id>.jpg, long edge 512 px, JPEG quality 90,
  whole directory <= 50 MB. The downscale is LOSSLESS FOR THIS PURPOSE, not
  bit-lossless: FaceNet consumes a 160x160 resize, so resolution above 512 px
  cannot influence the 512-d embedding, while 1024 px JPEGs would quadruple
  the cache. The cache exists so (a) generation runs ONCE, (b) it is
  chunkable — `ensure_face_image(customer_id)` is the per-customer unit, so a
  600 s ceiling kill leaves a valid partial cache the next run continues.
- embeddings: data/face_embeddings.npz (customer_id, embeddings 100x512
  float32, seed_rule) — a committed copy so downstream consumers (nb15, the
  runner) never need torch/facenet at all. `embed_customers()` reads the npz
  fast-path when it covers the requested customers, embeds only the missing
  ones otherwise, and merges the result back into the npz.

Embedding preprocessing (documented, standard facenet-pytorch recipe): cache
JPEG -> RGB -> resize 160x160 (LANCZOS) -> float /255 -> (x - 0.5) / 0.5
(i.e. the package's fixed_image_standardization, [0,1] -> [-1,1], the
normalization the vggface2 checkpoint was trained with). The output is
L2-normalized: vggface2 embeddings are compared cosine-style, and the unit
norm keeps the 512 columns scale-clean for any downstream consumer.

Why 512 raw columns and no reduction: a PCA would need FITTING, which would
turn append_features into a stateful, split-dependent transform and break the
label-free cached-extraction contract that makes supports_cv=True honest.
XGBoost handles 512 mostly-irrelevant columns fine at this scale (5302 rows),
the columns are unit-bounded after L2 normalization, and the pre-registered
expectation is a null — spending a reduction layer on it would manufacture
tuning freedom. The xgb-client matrix is ~116 encoded columns; 116 + 512 =
628 features on 5302 rows is well inside XGBoost's comfort zone.

Zero label access, stated structurally: the embedding is a function of the
customer id ALONE (hash -> z -> pixels -> FaceNet). fraud_flag never enters
any expression here; fold membership never changes a feature value, which is
exactly why the arm registers supports_cv=True.

check_dependencies() mirrors arms_timesfm.py: returns None when the arm can
run, else a short message — the runner's factories turn a non-None result
into an ArmSkipped row (fraud_pipeline.ArmSkipped), never a crash. The npz
fast-path means the arm can run (consumption-only) on machines with neither
torch nor facenet installed, as long as the committed npz is present.

Machine-local paths (no network, no pip package for the GAN): the NVIDIA
stylegan2-ada-pytorch repo is cloned at C:/sg2-ada and the FFHQ weights pkl
sits at C:/faces/ffhq.pkl. They are module-level constants WITH a fallback
probed in check_dependencies(); the custom CUDA kernels of that repo do not
compile against this torch, which silently falls back to the slow reference
implementation — same numbers, fewer fps.

Notebook API:
  from arms_face import (FACE_FEATURES, append_features, embed_customers,
                         ensure_face_image, face_path, generate_face_image,
                         check_dependencies)
  enriched_face = append_features(enriched)          # npz fast-path
  fresh = embed_customers(customers, use_cache=False)  # cache bypass (determinism check)
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# Machine-local constants (WHY: the StyleGAN2 repo is not pip-installable and
# the FFHQ pkl is a 364 MB local weight cache; both were provisioned once per
# machine by the phase setup. Fallback = existence probed by
# check_dependencies(), which reports a skip message instead of crashing.)
STYLEGAN_REPO_DIR = Path("C:/sg2-ada")
STYLEGAN_WEIGHTS_PATH = Path("C:/faces/ffhq.pkl")

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR.parent / "data"
FACE_DIR = DATA_DIR / "faces"
EMBEDDINGS_NPZ_PATH = DATA_DIR / "face_embeddings.npz"

# Cache budget: 100 JPEGs at 512 px / quality 90 land around 6-8 MB total;
# the assert keeps the committed cache inside the phase's 50 MB contract.
FACE_CACHE_MAX_BYTES = 50 * 1024 * 1024
FACE_LONG_EDGE = 512
FACE_JPEG_QUALITY = 90

# FaceNet input contract (vggface2 InceptionResnetV1): 160x160 RGB, [-1, 1].
FACENET_INPUT_SIZE = 160
FACENET_PRETRAINED = "vggface2"
EMB_DIM = 512

# StyleGAN2 sampling: truncation 0.7 is the FFHQ-quality default of the
# official generate.py; 'const' noise + fp32 keep synthesis free of RNG and
# fp16 rounding, which is what makes the seed rule byte-stable.
TRUNCATION_PSI = 0.7

# Seed-rule version tag: part of the hashed string so a future change of the
# rule cannot silently reuse old latents (domain separation, see docstring).
SEED_RULE_PREFIX = "face-arm-v1:"

# FaceNet weight file in the local torch-hub cache (facenet-pytorch 2.5.3
# resolves vggface2 through torch.hub.load_state_dict_from_url, which serves
# the cache FIRST and only downloads on a miss). check_dependencies probes
# this exact file so a generation run can never reach the network — the same
# no-download discipline as arms_text's local_files_only snapshot probe.
FACENET_WEIGHTS_FILENAME = "20180402-114759-vggface2.pt"

FACE_FEATURES = ["face_emb_%03d" % i for i in range(EMB_DIM)]


def _seed_rule_current():
    # The exact provenance string written into the npz (single source of
    # truth for save AND for the staleness gate at load).
    return (
        "sha256('%s' + customer_id)[:4] BE int (32-bit, RandomState range) "
        "-> RandomState.randn(z_dim), truncation_psi=%.1f, noise_mode=const, "
        "force_fp32" % (SEED_RULE_PREFIX, TRUNCATION_PSI)
    )

_GENERATOR = None
_FACENET = None
_EMBEDDING_CACHE = {}


def check_dependencies():
    # Probe WITHOUT network. Fast path: the committed npz alone covers
    # append_features (consumption needs no torch/facenet/GPU). Otherwise the
    # full generation+embedding stack is probed. Returns None when the arm can
    # run, else a short message — the runner's factory turns that into an
    # ArmSkipped row (both in build_features and make_model; build_features
    # runs first), never a crash.
    if EMBEDDINGS_NPZ_PATH.exists():
        return None
    missing = []
    for module in ("torch", "PIL", "facenet_pytorch"):
        try:
            __import__(module)
        except ImportError as exc:
            missing.append("%s (%s)" % (module, exc))
    if missing:
        return "; ".join(missing)
    import torch.hub  # lazy: torch already imported above

    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / FACENET_WEIGHTS_FILENAME
    if not weights_path.is_file():
        return (
            "FaceNet %s weights not in the local torch-hub cache (%s) — "
            "download once at setup; runtime must never fetch"
            % (FACENET_PRETRAINED, weights_path)
        )
    if not (STYLEGAN_REPO_DIR / "dnnlib").is_dir() or not (
        STYLEGAN_REPO_DIR / "legacy.py"
    ).is_file():
        return "stylegan2-ada-pytorch repo not found at %s" % STYLEGAN_REPO_DIR
    if not STYLEGAN_WEIGHTS_PATH.is_file():
        return "FFHQ weights not found at %s" % STYLEGAN_WEIGHTS_PATH
    return None


def customer_seed(customer_id):
    # D5 seed rule — the single source of image identity. SHA-256 so the seed
    # is stable across processes and platforms (hash() is salted per process
    # and must never be used for this). Truncated to the FIRST 4 BYTES: the
    # consumer is numpy's legacy MT19937 RandomState, whose seeding only
    # accepts 0..2**32-1 (a 64-bit int raises ValueError) — 32 bits of a
    # uniform digest is 100 customers' worth of collision headroom by miles.
    digest = hashlib.sha256((SEED_RULE_PREFIX + customer_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def face_path(customer_id, faces_dir=FACE_DIR):
    # Customer ids here are C/R + digits (verified: filesystem-safe), so the
    # id doubles as the file name — one id, one canonical image file.
    return Path(faces_dir) / ("%s.jpg" % customer_id)


def get_generator(device=None):
    # Load the StyleGAN2 G_ema once per process (registry path: generation in
    # ensure_face_image). sys.path surgery is required because the NVIDIA repo
    # is not a package on sys.path; the import surface is dnnlib + legacy only.
    global _GENERATOR
    if _GENERATOR is None:
        import sys

        import torch  # lazy: optional dependency

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assert (STYLEGAN_REPO_DIR / "legacy.py").is_file(), (
            "stylegan2 repo missing at %s" % STYLEGAN_REPO_DIR
        )
        assert STYLEGAN_WEIGHTS_PATH.is_file(), (
            "FFHQ weights missing at %s" % STYLEGAN_WEIGHTS_PATH
        )
        if str(STYLEGAN_REPO_DIR) not in sys.path:
            sys.path.insert(0, str(STYLEGAN_REPO_DIR))
        import dnnlib  # noqa: E402  (path inserted above, by design)
        import legacy  # noqa: E402

        torch.manual_seed(RANDOM_SEED)  # hygiene only: no torch RNG is
        # consumed below (noise buffers are 'const', baked into the pkl).
        # open_url on a plain local path just open()s it — no network fetch.
        with dnnlib.util.open_url(STYLEGAN_WEIGHTS_PATH.as_posix()) as handle:
            _GENERATOR = legacy.load_network_pkl(handle)["G_ema"].to(device)
    return _GENERATOR


def generate_face_image(customer_id, out_path):
    # Synthesize the customer's face from the D5 seed and cache it as JPEG.
    # The heavy imports live here: importing this module must never require
    # torch/facenet/stylegan (module-chain rule of the other arms_* modules).
    import torch  # lazy: optional dependency

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = get_generator(device)
    label = torch.zeros([1, generator.c_dim], device=device)
    z = torch.from_numpy(
        np.random.RandomState(customer_seed(customer_id)).randn(1, generator.z_dim)
    ).to(device)
    with torch.no_grad():
        image = generator.synthesis(
            generator.mapping(z, label, truncation_psi=TRUNCATION_PSI),
            noise_mode="const",
            force_fp32=True,
        )
    # Official generate.py convention: network output is [-1, 1] RGB.
    pixels = (
        image.add(1.0).div(2.0).clamp(0.0, 1.0).mul(255.0)
        .to(torch.uint8).permute(0, 2, 3, 1)[0].cpu().numpy()
    )
    from PIL import Image  # lazy: optional dependency

    pil = Image.fromarray(pixels, mode="RGB")
    pil.thumbnail((FACE_LONG_EDGE, FACE_LONG_EDGE), Image.LANCZOS)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path, format="JPEG", quality=FACE_JPEG_QUALITY)
    return out_path


def ensure_face_image(customer_id, faces_dir=FACE_DIR):
    # Generate-or-load, per customer — the chunkable cache unit. Returns
    # (path, generated): generated=True only when synthesis actually ran, so
    # callers can report cache hits vs misses and a 600 s kill never loses a
    # half-written cache (each file is written whole, one at a time).
    path = face_path(customer_id, faces_dir)
    if path.is_file():
        return path, False
    return generate_face_image(customer_id, path), True


def _get_facenet(device):
    # FaceNet once per process, eval mode (no dropout -> deterministic),
    # weights served from the local torch-hub cache — never downloaded here.
    import torch  # lazy: optional dependency
    from facenet_pytorch import InceptionResnetV1  # lazy: optional dependency

    global _FACENET
    if _FACENET is None:
        torch.manual_seed(RANDOM_SEED)  # no RNG consumed in eval-mode inference
        model = InceptionResnetV1(pretrained=FACENET_PRETRAINED, classify=False)
        _FACENET = model.to(device).eval()
    return _FACENET


def _image_tensor(path, device):
    # JPEG -> (1, 3, 160, 160) float tensor in [-1, 1] — the standard
    # facenet-pytorch preprocessing (fixed_image_standardization equivalent).
    import torch  # lazy: optional dependency
    from PIL import Image  # lazy: optional dependency

    pil = Image.open(path).convert("RGB").resize(
        (FACENET_INPUT_SIZE, FACENET_INPUT_SIZE), Image.LANCZOS
    )
    pixels = np.asarray(pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy((pixels - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def embed_customer(customer_id, use_cache=True, device=None):
    # 512-d embedding for ONE customer. Thin wrapper over embed_customers —
    # same resolution order, same process cache keyed by the id (the brief's
    # per-customer cache: the embedding depends on the id alone). use_cache=
    # False forces a fresh FaceNet pass — the determinism check's bypass.
    if use_cache and customer_id in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[customer_id]
    frame = embed_customers([customer_id], use_cache=use_cache, device=device)
    embedding = frame.loc[customer_id].to_numpy(dtype=np.float32)
    if use_cache:
        _EMBEDDING_CACHE[customer_id] = embedding
    return embedding


def embed_customers(customer_ids, use_cache=True, device=None):
    # DataFrame (index=customer_id, columns=FACE_FEATURES, float32) for the
    # requested customers. Per-customer resolution order: process cache
    # (keyed by customer identity) -> committed npz (no GPU pass, the
    # downstream fast path) -> a fresh FaceNet pass on the cached JPEG.
    # Freshly computed customers are merged back into the npz so the next
    # process never re-embeds them.
    customer_ids = list(customer_ids)
    npz_frame = load_embeddings() if use_cache else None
    if npz_frame is not None and set(customer_ids).issubset(npz_frame.index):
        return npz_frame.loc[customer_ids]
    import torch  # lazy: optional dependency

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fresh_ids = [
        c for c in customer_ids
        if not (
            use_cache
            and (c in _EMBEDDING_CACHE or (npz_frame is not None and c in npz_frame.index))
        )
    ]
    fresh = {}
    if fresh_ids:
        model = _get_facenet(device)
        tensors = torch.cat(
            [_image_tensor(ensure_face_image(c)[0], device) for c in fresh_ids]
        )
        with torch.no_grad():
            batch = model(tensors).cpu().numpy()
        for customer_id, row in zip(fresh_ids, batch):
            unit = (row / np.linalg.norm(row)).astype(np.float32)
            fresh[customer_id] = unit
            if use_cache:
                _EMBEDDING_CACHE[customer_id] = unit
    rows = []
    for customer_id in customer_ids:
        if customer_id in fresh:
            rows.append(fresh[customer_id])
        elif use_cache and customer_id in _EMBEDDING_CACHE:
            rows.append(_EMBEDDING_CACHE[customer_id])
        else:
            assert npz_frame is not None and customer_id in npz_frame.index, (
                "no embedding source for customer %s" % customer_id
            )
            rows.append(npz_frame.loc[customer_id].to_numpy(dtype=np.float32))
    result = pd.DataFrame(
        np.asarray(rows, dtype=np.float32), index=customer_ids, columns=FACE_FEATURES
    )
    assert np.isfinite(result.to_numpy()).all(), "non-finite face embedding"
    if fresh:
        _merge_fresh_into_npz(fresh)
    return result


def _read_npz(path):
    # Shared npz reader: returns (ids, embeddings float32, seed_rule str) or
    # None when the artifact does not exist. A legacy npz without the
    # seed_rule field reads as "" — the gate below treats it as stale.
    if not Path(path).is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        seed_rule = str(data["seed_rule"][0]) if "seed_rule" in data.files else ""
        return list(data["customer_id"]), data["embeddings"].astype(np.float32), seed_rule


def load_embeddings(path=EMBEDDINGS_NPZ_PATH):
    # The committed artifact as a DataFrame (index=customer_id), or None when
    # it does not exist yet (first generation run writes it) or when it is
    # STALE: a seed_rule that does not match the current rule means the file
    # was produced under a different latents rule (e.g. the 64->32-bit fix)
    # and must be regenerated, never consumed — the same trust-gate class as
    # arms_demo's corpus digest (review r1). Self-heals: the next fresh pass
    # overwrites the file with the current rule.
    read = _read_npz(path)
    if read is None:
        return None
    ids, embeddings, seed_rule = read
    if seed_rule != _seed_rule_current():
        return None
    frame = pd.DataFrame(embeddings, index=ids, columns=FACE_FEATURES)
    return frame.sort_index()


def _merge_fresh_into_npz(fresh, path=EMBEDDINGS_NPZ_PATH):
    # Existing npz rows + freshly embedded ones; a fresh value always wins for
    # the customers actually recomputed (determinism checks write back too).
    existing = load_embeddings(path)
    fresh_frame = pd.DataFrame.from_dict(fresh, orient="index", columns=FACE_FEATURES)
    if existing is None:
        merged = fresh_frame
    else:
        merged = pd.concat([existing, fresh_frame])
        merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    save_embeddings(merged, path)
    return merged


def save_embeddings(frame, path=EMBEDDINGS_NPZ_PATH):
    # Persist the committed embedding artifact: customer_id + 100x512 float32
    # + the seed rule, so the exact provenance rides inside the file.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        customer_id=np.asarray(frame.index, dtype="U"),
        embeddings=frame[FACE_FEATURES].to_numpy(dtype=np.float32),
        seed_rule=np.asarray([_seed_rule_current()], dtype="U"),
    )


def append_features(frame, use_cache=True):
    """frame + the FACE_FEATURES columns, one embedding broadcast per customer.

    The embedding is a function of the customer id ALONE (hash -> pixels ->
    FaceNet), so broadcasting is a plain per-customer constant join: every row
    of a customer carries that customer's 512-d vector. Zero label access —
    fraud_flag is never read, and fold membership cannot change a value, which
    is why the arm ships supports_cv=True. Returns a NEW frame; the input is
    never mutated. use_cache=False bypasses both the process cache and the npz
    (forces a full FaceNet pass — the determinism check's lever).
    """
    assert "customer" in frame.columns, "frame needs the customer column"
    embeddings = embed_customers(frame["customer"].unique(), use_cache=use_cache)
    assert set(frame["customer"].unique()).issubset(embeddings.index), (
        "embedding table missing customers"
    )
    features = embeddings.loc[frame["customer"].to_numpy()]
    features.index = frame.index
    return pd.concat([frame, features], axis=1)


def assert_face_cache_within_budget(faces_dir=FACE_DIR):
    # Cache-size invariant (the phase's 50 MB contract), checked by nb13 after
    # generation so an accidental quality/size regression fails loudly.
    total = sum(p.stat().st_size for p in Path(faces_dir).glob("*.jpg"))
    assert total <= FACE_CACHE_MAX_BYTES, (
        "face cache %d bytes exceeds the %d byte budget" % (total, FACE_CACHE_MAX_BYTES)
    )
    return total
