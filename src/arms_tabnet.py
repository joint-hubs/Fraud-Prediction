"""TabNet arm (FOC-175 F3): pytorch-tabnet on the base+client feature set.

Implements the `tabnet` arm of the unified runner (fraud_pipeline.py):
TabNetClassifier (pytorch-tabnet 4.1.0) trained on the SAME 116-column
base+client matrix as the xgb-client / gbdt-ensemble arms (LightGBM-safe
column names via arms_gbdt.sanitize_feature_names so the arms' feature
handling stays consistent; values and column order unchanged). The dummy
columns are consumed as plain floats — the shared matrix is used as-is, no
TabNet-native categorical embeddings (cat_idxs/cat_dims) on top.

House discipline mirrors fraud_pipeline.FIXED_XGB_PARAMS / arms_gbdt:

- fixed hyperparameters (library defaults for the architecture: n_d=n_a=8,
  n_steps=3, gamma=1.3, sparsemax masks, Adam lr 2e-2) and early stopping on
  an internal stratified validation carve (25% of the fitting rows, seed 42 —
  the same shape as the runner's threshold carve) with eval_metric 'auc',
  patience 10, max_epochs 60. run_arm_on_split hands fit() only its fitting
  carve, so the early-stopping split must be cut inside fit();
- fixed seeds: constructor seed=42 plus torch.manual_seed / np.random.seed /
  torch.backends.cudnn.deterministic=True (benchmark off) re-pinned at every
  fit(). GPU determinism is attempted, not guaranteed — the notebook fits the
  same model repeatedly and reports the observed deviation, if any;
- device: CUDA when torch.cuda.is_available() else CPU, resolved in the
  factory and recorded on the fitted model (device_name).

Imbalance handling (honest disclosure): pytorch-tabnet 4.1.0 has no
loss-level class weight — there is no scale_pos_weight equivalent. Its only
documented imbalance mechanism for classifiers is fit(weights=1), which swaps
shuffle for a WeightedRandomSampler with inverse-frequency class weights
(library-managed minority oversampling, pytorch_tabnet.utils.create_sampler).
This arm uses exactly that documented knob — the rare-class emphasis then
matches the scale_pos_weight / class_weights the other arms bake in — and
NOTHING beyond it: no SMOTE, no manual resampling. The frozen-threshold
tuning on the runner's validation carve stays the threshold-side compensator.

Dependencies are optional on other machines: nothing imports torch or
pytorch_tabnet at module import time. check_dependencies() probes them so the
runner's factory can turn a missing dependency into a SKIPPED row
(fraud_pipeline.ArmSkipped) instead of a crash.

CV capability: per-fold GPU clones would each re-carve, re-seed and re-run
early stopping for a few seconds a fold — possible, but not worth it at this
scale (and the arm's evidence lives on the customer-disjoint test axes
anyway); it registers supports_cv=False ([no-cv] in the runner).

Notebook API:
  from arms_tabnet import fit_params, make_tabnet, tabnet_params
  model = make_tabnet(y_fit)
  model.fit(X_fit, y_fit)
  proba = model.predict_proba(X_test)[:, 1]
  M_explain, masks = model.explain_matrix(X_fit, normalize=True)
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def check_dependencies():
    # Probe-import the heavy libraries. Returns None when everything is
    # installed, otherwise a short message naming the missing module(s) — the
    # runner's factory turns a non-None result into an ArmSkipped row.
    missing = []
    for module in ("torch", "pytorch_tabnet"):
        try:
            __import__(module)
        except ImportError as exc:
            missing.append("%s (%s)" % (module, exc))
    return "; ".join(missing) if missing else None


def tabnet_params(device_name):
    # Constructor params. Architecture stays at the library defaults (n_d=n_a=8
    # steps=3 gamma=1.3 lambda_sparse=1e-3 sparsemax, Adam lr 2e-2 — ~4k rows
    # does not justify a search); the pins are seed, device and quiet logging.
    return dict(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        seed=RANDOM_SEED,
        device_name=device_name,
        verbose=0,  # the runner owns the printed run lines
    )


def fit_params():
    # fit()-time params: early stopping on the internal carve + the documented
    # imbalance sampler (weights=1 -> inverse-frequency WeightedRandomSampler;
    # see the module docstring for why nothing beyond this knob is applied).
    return dict(
        max_epochs=60,
        patience=10,
        batch_size=256,
        virtual_batch_size=128,
        weights=1,
    )


def make_tabnet(y_fit):
    # Registry factory (fraud_pipeline._make_tabnet). y_fit is accepted for the
    # registry's make_model(y_fit) contract but unused: the weights=1 sampler
    # and the internal carve both derive class frequencies from the actual
    # fitting rows inside fit(), which is the honest source (the carve fit()
    # receives is already the smallest training set in the protocol).
    import torch  # lazy: optional dependency

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabNetModel(tabnet_params=tabnet_params(device), fit_params=fit_params())


class TabNetModel(BaseEstimator):
    """Sklearn-style wrapper around TabNetClassifier (pytorch-tabnet 4.1.0).

    fit() pins the global seeds (torch / numpy / cuDNN deterministic), cuts the
    stratified early-stopping carve from the fitting rows it receives, and fits
    with eval_metric 'auc' + patience. predict_proba() returns the (n_samples,
    2) array the runner and funs.evaluateModel expect — column 1 is the fraud
    score. explain_matrix() exposes the library's explain() attention masks for
    the notebook's feature-group aggregation. Constructor params are plain
    dicts, so the estimator clones without carrying fitted state (the sklearn
    get_params() contract, same as arms_gbdt.GBDTSoftVoteEnsemble).
    """

    def __init__(self, tabnet_params=None, fit_params=None):
        self.tabnet_params = tabnet_params
        self.fit_params = fit_params

    @staticmethod
    def _values(X):
        # float32 matrix — TabNet consumes numpy only. The dummies (bool /
        # uint8) and the two client numerics (float64) all cast losslessly.
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        import torch  # lazy: optional dependency
        from pytorch_tabnet.tab_model import TabNetClassifier  # lazy

        X_arr = self._values(X)
        y_arr = np.asarray(y).ravel().astype(int)

        # Seed discipline, re-pinned per fit (also covers repeat fits inside
        # one kernel and the library's own seed handling): deterministic cuDNN
        # kernel selection, benchmark autotune off. Global torch state — set
        # here so every fit path (runner, CV clones, notebook refits) shares it.
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Early stopping needs an eval set, but run_arm_on_split hands fit()
        # only its fitting carve: cut the validation split here — stratified,
        # 25%, seed 42, the same shape as the runner's threshold carve. Both
        # sides keep both classes by construction (stratify).
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_arr, y_arr, test_size=0.25, random_state=RANDOM_SEED, stratify=y_arr
        )
        self.carve_ = {
            "fit_rows": int(len(y_fit)),
            "fit_positives": int(y_fit.sum()),
            "val_rows": int(len(y_val)),
            "val_positives": int(y_val.sum()),
        }

        self.feature_names_ = (
            [str(c) for c in X.columns]
            if hasattr(X, "columns")
            else [str(i) for i in range(X_arr.shape[1])]
        )

        params = dict(self.fit_params or {})
        self.device_name_ = dict(self.tabnet_params or {}).get("device_name", "auto")
        self.model_ = TabNetClassifier(**dict(self.tabnet_params or {}))
        self.model_.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["auc"],
            weights=params.get("weights", 0),
            max_epochs=params.get("max_epochs", 60),
            patience=params.get("patience", 10),
            batch_size=params.get("batch_size", 256),
            virtual_batch_size=params.get("virtual_batch_size", 128),
            drop_last=False,  # default True would drop the tail batch (~5% of rows)
            compute_importance=False,  # explain() runs on demand in the notebook
        )
        self.val_auc_history_ = [float(v) for v in self.model_.history["val_auc"]]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        # TabNetClassifier.predict_proba returns (n_samples, 2) for binary
        # targets; np.asarray keeps the contract even if a list sneaks in.
        return np.asarray(self.model_.predict_proba(self._values(X)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def explain_matrix(self, X, normalize=True):
        # Attention-mask explanation from the underlying network:
        # M_explain (n_samples, n_features) aggregated across decision steps —
        # row-normalized when normalize=True so per-row shares sum to 1 — and
        # masks {step: (n_samples, n_features)} per step.
        M_explain, masks = self.model_.explain(self._values(X), normalize=normalize)
        return np.asarray(M_explain), masks
