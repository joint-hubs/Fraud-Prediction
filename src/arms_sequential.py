"""Sequential arms (FOC-175 F3): per-customer LSTM and Transformer encoders.

Implements the `sequential-lstm` and `sequential-transformer` arms of the
unified runner (fraud_pipeline.py). Both consume the SAME per-customer
transaction sequences and differ only in the sequence encoder; the hypothesis
under test is "unusual for THIS customer": a transaction is scored against
what its own customer did strictly before it, not against a global population.

Sequence construction (shared `_pack_sequences`, one scheme for both arms):

- rows are stably sorted by (customer, timestamp, original row order); each
  customer's run is one sequence, steps in time order;
- per-step inputs: the 9 factor categoricals (type, ccy, customer_country,
  counterparty_country, customer_type, weekday, month, quarter, hour) each
  embedded via a small learned table, plus amount_eur as log1p + standardize
  (scaler fitted on the fitting rows only). NO label-derived feature anywhere
  in a sequence — fraud_flag never enters the step tensors (leakage).

Past-only discipline (the leakage bar for sequence models):

- LSTM: transaction i is scored from its own step embedding (the query)
  concatenated with the recurrent state AFTER step i-1 (strictly earlier
  steps as context); step 0 of a sequence gets a zero context state.
- Transformer: causal self-attention keeping the diagonal (j <= i) — step i's
  own attributes enter as its query (and its single self-value), every other
  attended step is strictly earlier; no future step ever contributes.
- Inference mirrors training: predict_proba() builds sequences from the rows
  it receives ALONE, so on the customer-disjoint axes a test customer's
  earlier test transactions serve as inference-time context only (no
  fitting, no labels) and the model never saw the customer in training.

Splitting / protocol compliance:

- run_arm_on_split hands fit() only the runner's fitting carve (a ROW-level
  stratified split, customer-mixed — the runner owns that carve). The
  early-stopping split is cut INSIDE fit() at the CUSTOMER level (stratified
  on has-fraud, seed 42): a row-level cut would fragment per-customer
  sequences and leak a fit row into a val sequence's context.
- Imbalance handling (honest disclosure, no silent oversampling): a pos_weight
  = n_negative / n_positive on the training steps inside BCEWithLogitsLoss —
  the loss-weight equivalent of the XGB scale_pos_weight every tree arm
  bakes in. Nothing beyond it; the frozen-threshold tuning on the runner's
  validation carve stays the threshold-side compensator.

Determinism: fixed seeds re-pinned at every fit (torch / numpy,
cudnn.deterministic=True, benchmark off) plus a dedicated seeded RNG for the
per-epoch customer permutation. GPU determinism is attempted, not guaranteed —
the notebook refits repeatedly and reports the observed deviation.

Cost: d_model 32-64 scale models on <=94-step sequences (5302 txns / 100
customers), max 40 epochs with early stopping — each fit is seconds on CUDA.

CV capability: a fold clone would re-pack, re-carve and re-run early stopping
per fold; the arm's evidence lives on the customer-disjoint test axes, so it
registers supports_cv=False ([no-cv] in the runner).

Dependencies are optional on other machines: nothing imports torch at module
import time. check_dependencies() probes it so the runner's factories can turn
a missing dependency into a SKIPPED row (fraud_pipeline.ArmSkipped).

Notebook API:
  from arms_sequential import make_sequential, build_sequence_frame
  model = make_sequential("lstm", y_fit)
  model.fit(X_fit, y_fit)
  proba = model.predict_proba(X_test)[:, 1]
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# Per-step factor features (the brief's contract): 9 embedded categoricals +
# the scaled amount. The sequence keys (customer, timestamp) ride along for
# ordering only; no label-derived column is ever included.
SEQ_CATEGORICAL = [
    "type",
    "ccy",
    "customer_country",
    "counterparty_country",
    "customer_type",
    "weekday",
    "month",
    "quarter",
    "hour",
]
SEQ_NUMERIC = ["amount_eur"]
SEQ_FRAME_COLUMNS = SEQ_CATEGORICAL + SEQ_NUMERIC + ["customer", "timestamp"]

# Per-categorical embedding width; vocabs here are tiny (3-24 values), so a
# fixed 8-dim table per feature keeps the step encoder small and uniform.
EMB_DIM = 8
# Positional-embedding budget for the Transformer; the longest per-customer
# sequence on this table is 94 transactions (nb12 sequence-shape cell).
MAX_STEPS = 128


def build_sequence_frame(enriched):
    """The frame the sequential estimators consume: the per-step factor
    columns plus the sequence keys, index preserved (the runner slices it by
    train/test indices). fraud_flag is deliberately absent."""
    missing = [c for c in SEQ_FRAME_COLUMNS if c not in enriched.columns]
    assert not missing, "sequence frame missing columns: %s" % missing
    return enriched[SEQ_FRAME_COLUMNS]


def check_dependencies():
    # Probe-import the heavy library. Returns None when installed, otherwise a
    # short message — the runner's factory turns a non-None result into an
    # ArmSkipped row.
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return "torch (%s)" % exc
    return None


def lstm_params():
    # Fixed hyperparameters (no search at ~3k fitting rows): 1 recurrent layer,
    # hidden 64, small MLP head. Constructor-plain dict (sklearn clone contract).
    return dict(
        hidden_size=64,
        num_layers=1,
        emb_dim=EMB_DIM,
        mlp_hidden=32,
        dropout=0.1,
    )


def transformer_params():
    # Fixed hyperparameters: 2 encoder layers, d_model 64, 4 heads, FFN 128,
    # learned positional embeddings (MAX_STEPS covers the 94-step maximum).
    return dict(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        emb_dim=EMB_DIM,
        mlp_hidden=32,
        dropout=0.1,
        max_steps=MAX_STEPS,
    )


def fit_params():
    # fit()-time params: few epochs, early stopping on the internal
    # customer-level carve's PR-AUC, customer mini-batches.
    return dict(
        max_epochs=40,
        patience=8,
        batch_customers=32,
        lr=1e-3,
    )


def make_sequential(kind, y_fit):
    # Registry factory (fraud_pipeline._make_sequential_*). y_fit is accepted
    # for the registry's make_model(y_fit) contract but unused: the pos_weight
    # and the internal carve both derive class frequencies from the actual
    # fitting rows inside fit() (the honest source — see arms_tabnet).
    import torch  # lazy: optional dependency

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_params = lstm_params() if kind == "lstm" else transformer_params()
    return SequenceModel(
        kind=kind,
        model_params=model_params,
        fit_params=fit_params(),
        device_name=device,
    )


class _StepTable:
    """Label-free per-step encoding, fitted on the rows fit() receives only.

    Categoricals are str-cast into vocabularies with index 0 reserved for
    unseen values (a zero embedding vector — honest no-information handling
    for a category the fit never saw); amount_eur is log1p + standardized
    with the fit rows' mean/std, re-used verbatim at predict time.
    """

    def fit(self, frame):
        self.vocab_ = {}
        for col in SEQ_CATEGORICAL:
            values = frame[col].astype(str)
            self.vocab_[col] = {
                value: index + 1
                for index, value in enumerate(sorted(values.unique()))
            }
        amounts = np.log1p(frame[SEQ_NUMERIC[0]].to_numpy(dtype=float))
        self.amount_mean_ = float(amounts.mean())
        self.amount_std_ = float(amounts.std()) or 1.0
        return self

    def codes(self, frame):
        return {
            col: frame[col]
            .astype(str)
            .map(self.vocab_[col])
            .fillna(0)
            .to_numpy(dtype=np.int64)
            for col in SEQ_CATEGORICAL
        }

    def amount(self, frame):
        amounts = np.log1p(frame[SEQ_NUMERIC[0]].to_numpy(dtype=float))
        return (amounts - self.amount_mean_) / self.amount_std_


def _pack_sequences(frame, step_table):
    """Pack transaction rows into per-customer step tensors.

    Rows are stably sorted by (customer, timestamp, original row order) — the
    stable sort makes the sequence order — and therefore every feature —
    deterministic under row shuffles. Each customer's run becomes one sequence,
    right-padded to the longest. Returns:

      positions  (n_customers, T) int   original row positions, -1 = padding
      x_cat      (n_customers, T, 9) int64  categorical codes (0 unseen/pad)
      x_amt      (n_customers, T) float32  scaled amount_eur
      pad_mask   (n_customers, T) bool  True = real step
      customers  (n_customers,)    the customer id of each sequence
    """
    work = pd.DataFrame(
        {
            "customer": frame["customer"].to_numpy(),
            "timestamp": frame["timestamp"].to_numpy(),
            "pos": np.arange(len(frame)),
        }
    ).sort_values(["customer", "timestamp", "pos"], kind="stable")
    customers_col = work["customer"].to_numpy()
    boundary = np.r_[True, customers_col[1:] != customers_col[:-1]]
    run_id = np.cumsum(boundary) - 1
    starts = np.flatnonzero(boundary)
    n_customers = len(starts)
    lengths = np.bincount(run_id, minlength=n_customers)
    max_len = int(lengths.max())

    positions = -np.ones((n_customers, max_len), dtype=int)
    within = np.arange(len(work)) - np.repeat(starts, lengths)
    positions[run_id, within] = work["pos"].to_numpy()

    pad_mask = positions >= 0
    safe_pos = np.where(pad_mask, positions, 0)

    codes = step_table.codes(frame)
    x_cat = np.stack(
        [codes[col][safe_pos] * pad_mask for col in SEQ_CATEGORICAL], axis=-1
    ).astype(np.int64)
    x_amt = (step_table.amount(frame)[safe_pos] * pad_mask).astype(np.float32)
    return positions, x_cat, x_amt, pad_mask, customers_col[starts]


class SequenceModel(BaseEstimator):
    """Sklearn-style wrapper over the two sequence encoders.

    fit() pins the global seeds, builds the per-step encoding + sequences from
    the rows it receives, cuts the customer-level early-stopping carve, and
    trains with BCEWithLogitsLoss(pos_weight) + early stopping on the carve's
    PR-AUC. predict_proba() re-packs the rows it receives with the FITTED
    step table (unseen categories -> the zero vector) and returns the (n, 2)
    array the runner expects, aligned to the input row order. Constructor
    params are plain data, so the estimator clones without fitted state.
    """

    def __init__(self, kind, model_params=None, fit_params=None, device_name="auto"):
        self.kind = kind
        self.model_params = model_params
        self.fit_params = fit_params
        self.device_name = device_name

    # -- torch modules (defined lazily: torch is an optional dependency) ----

    def _build_net(self, vocab_sizes):
        import torch  # lazy: optional dependency
        from torch import nn

        kind = self.kind
        params = dict(self.model_params or {})

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.ModuleList(
                    [
                        nn.Embedding(size + 1, params["emb_dim"], padding_idx=0)
                        for size in vocab_sizes
                    ]
                )
                input_dim = params["emb_dim"] * len(vocab_sizes) + 1
                if kind == "lstm":
                    self.seq = nn.LSTM(
                        input_dim,
                        params["hidden_size"],
                        num_layers=params["num_layers"],
                        batch_first=True,
                        dropout=params["dropout"] if params["num_layers"] > 1 else 0.0,
                    )
                    rep_dim = params["hidden_size"]
                else:
                    self.input_proj = nn.Linear(input_dim, params["d_model"])
                    self.pos_emb = nn.Embedding(params["max_steps"], params["d_model"])
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=params["d_model"],
                        nhead=params["nhead"],
                        dim_feedforward=params["dim_feedforward"],
                        dropout=params["dropout"],
                        batch_first=True,
                    )
                    self.seq = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=params["num_layers"],
                        enable_nested_tensor=False,
                    )
                    rep_dim = params["d_model"]
                self.head = nn.Sequential(
                    nn.Linear(input_dim + rep_dim, params["mlp_hidden"]),
                    nn.ReLU(),
                    nn.Dropout(params["dropout"]),
                    nn.Linear(params["mlp_hidden"], 1),
                )

            def forward(self, x_cat, x_amt, pad_mask):
                embs = [
                    emb(x_cat[:, :, k]) for k, emb in enumerate(self.embeddings)
                ]
                steps = torch.cat(embs + [x_amt.unsqueeze(-1)], dim=-1)
                if kind == "lstm":
                    out, _ = self.seq(steps)
                    # Strictly-past context: state after step i-1 (zero at i=0).
                    context = torch.zeros_like(out)
                    context[:, 1:] = out[:, :-1]
                else:
                    n_steps = steps.shape[1]
                    pos = torch.arange(n_steps, device=steps.device)
                    hidden = self.input_proj(steps) + self.pos_emb(pos).unsqueeze(0)
                    # Causal mask keeping the diagonal: query i may attend to
                    # keys j <= i — its own attributes plus strictly earlier
                    # steps, never a future one. Bool convention: True =
                    # blocked; padded keys are blocked via src_key_padding_mask
                    # (position 0 is always real, so no query row is empty).
                    causal = torch.triu(
                        torch.ones(
                            n_steps, n_steps, dtype=torch.bool, device=steps.device
                        ),
                        diagonal=1,
                    )
                    out = self.seq(
                        hidden, mask=causal, src_key_padding_mask=~pad_mask
                    )
                    context = out
                return self.head(
                    torch.cat([steps, context], dim=-1)
                ).squeeze(-1)

        return _Net()

    # -- sklearn-style API ---------------------------------------------------

    def fit(self, X, y):
        import torch  # lazy: optional dependency
        from torch import nn

        assert list(X.columns) == SEQ_FRAME_COLUMNS, (
            "sequential arms expect exactly the sequence frame columns"
        )
        # Seed discipline, re-pinned per fit (covers repeat fits inside one
        # kernel and every fit path — runner, notebook refits): deterministic
        # cuDNN kernel selection, benchmark autotune off.
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        frame = X
        y_arr = np.asarray(y).ravel().astype(float)
        self.step_table_ = _StepTable().fit(frame)
        positions, x_cat, x_amt, pad_mask, customers = _pack_sequences(
            frame, self.step_table_
        )
        n_customers, max_len = pad_mask.shape
        assert max_len <= MAX_STEPS, (
            "sequence longer than the positional budget (%d > %d)" % (max_len, MAX_STEPS)
        )

        step_labels = y_arr[np.where(pad_mask, positions, 0)] * pad_mask
        device = torch.device(self.device_name)

        # Early stopping carve, cut at the CUSTOMER level (a row-level cut
        # would fragment sequences and leak a fit row into a val context).
        # Stratified on has-fraud so both sides see positives.
        has_fraud = np.array(
            [
                step_labels[i][pad_mask[i]].max() > 0 if pad_mask[i].any() else 0.0
                for i in range(n_customers)
            ]
        )
        stratify = None
        counts = np.bincount(has_fraud.astype(int))
        if counts.min() >= 2:
            stratify = has_fraud
        fit_pos, val_pos = train_test_split(
            np.arange(n_customers),
            test_size=0.25,
            random_state=RANDOM_SEED,
            stratify=stratify,
        )
        train_steps = int(pad_mask[fit_pos].sum())
        train_positives = int(step_labels[fit_pos][pad_mask[fit_pos]].sum())
        val_steps = int(pad_mask[val_pos].sum())
        val_positives = int(step_labels[val_pos][pad_mask[val_pos]].sum())
        assert val_positives >= 1, "early-stopping carve has no positive step"
        # Imbalance handling (disclosed): loss weight = neg/pos over the
        # training steps. Nothing beyond it — no oversampling anywhere.
        self.pos_weight_ = (train_steps - train_positives) / max(train_positives, 1)
        self.carve_ = {
            "fit_customers": int(len(fit_pos)),
            "fit_steps": train_steps,
            "fit_positives": train_positives,
            "val_customers": int(len(val_pos)),
            "val_steps": val_steps,
            "val_positives": val_positives,
            "pos_weight": float(self.pos_weight_),
        }

        vocab_sizes = [len(self.step_table_.vocab_[c]) for c in SEQ_CATEGORICAL]
        net = self._build_net(vocab_sizes).to(device)
        params = dict(self.fit_params or {})
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight_], device=device)
        )
        optimizer = torch.optim.Adam(
            net.parameters(), lr=float(params.get("lr", 1e-3))
        )

        x_cat_t = torch.from_numpy(x_cat).to(device)
        x_amt_t = torch.from_numpy(x_amt).to(device)
        pad_t = torch.from_numpy(pad_mask).to(device)
        labels_t = torch.from_numpy(step_labels.astype(np.float32)).to(device)
        fit_idx = torch.from_numpy(np.sort(fit_pos)).to(device)
        val_idx = torch.from_numpy(np.sort(val_pos)).to(device)
        rng = np.random.RandomState(RANDOM_SEED)  # permutation RNG, fit-local
        batch = int(params.get("batch_customers", 32))

        history = []
        best_ap, best_state, epochs_since_best = -1.0, None, 0
        for epoch in range(int(params.get("max_epochs", 40))):
            net.train()
            perm = rng.permutation(len(fit_pos))
            for start in range(0, len(perm), batch):
                batch_idx = fit_idx[torch.from_numpy(perm[start : start + batch])]
                logits = net(x_cat_t[batch_idx], x_amt_t[batch_idx], pad_t[batch_idx])
                real = pad_t[batch_idx]
                loss = criterion(logits[real], labels_t[batch_idx][real])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                val_logits = net(x_cat_t[val_idx], x_amt_t[val_idx], pad_t[val_idx])
            val_real = pad_t[val_idx]
            val_p = (
                torch.sigmoid(val_logits)[val_real].detach().cpu().numpy()
            )
            val_y = step_labels[val_pos][pad_mask[val_pos]]
            epoch_ap = float(average_precision_score(val_y, val_p))
            history.append(epoch_ap)
            if epoch_ap > best_ap + 1e-12:
                best_ap, epochs_since_best = epoch_ap, 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in net.state_dict().items()
                }
            else:
                epochs_since_best += 1
                if epochs_since_best >= int(params.get("patience", 8)):
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

        self.model_ = net
        self.device_name_ = str(device)
        self.history_ = [float(v) for v in history]
        self.best_val_pr_auc_ = float(best_ap)
        self.best_epoch_ = int(np.argmax(history)) + 1 if history else 0
        self.epochs_run_ = len(history)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import torch  # lazy: optional dependency

        assert list(X.columns) == SEQ_FRAME_COLUMNS, (
            "sequential arms expect exactly the sequence frame columns"
        )
        device = torch.device(self.device_name_)
        positions, x_cat, x_amt, pad_mask, _ = _pack_sequences(X, self.step_table_)
        scores = np.full(len(X), np.nan)
        chunk = 256  # customers per forward pass (generality; 100 here)
        with torch.no_grad():
            for start in range(0, len(positions), chunk):
                stop = start + chunk
                logits = self.model_(
                    torch.from_numpy(x_cat[start:stop]).to(device),
                    torch.from_numpy(x_amt[start:stop]).to(device),
                    torch.from_numpy(pad_mask[start:stop]).to(device),
                )
                proba = torch.sigmoid(logits).cpu().numpy()
                real = pad_mask[start:stop]
                scores[positions[start:stop][real]] = proba[real]
        assert np.isfinite(scores).all(), "every row must be scored exactly once"
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
