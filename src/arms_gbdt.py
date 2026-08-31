"""GBDT-ensemble arm (FOC-175 F3): XGB + LightGBM + CatBoost soft vote.

Implements the `gbdt-ensemble` arm of the unified runner (fraud_pipeline.py):
three gradient-boosted-tree classifiers trained on the SAME feature matrix and
combined by equal-weight soft voting (the mean predicted fraud probability).
The arm's build_features hands members the base+client matrix with
sanitize_feature_names applied — LightGBM bans JSON special characters in
feature names and the amount_eur_bucket interval labels carry a comma.

House discipline mirrors fraud_pipeline.xgb_params / FIXED_XGB_PARAMS:

- fixed hyperparameters everywhere, mirroring the XGB set where the parameter
  exists per library (learning_rate 0.05, depth 6, 200 trees); each library
  keeps its own default regularization, noted inline;
- rare-class weighting computed from the fitting carve's y (member_params):
  scale_pos_weight for XGB/LightGBM, class_weights for CatBoost — one weight
  baked into the constructor params, so cross_validate_model's per-fold clones
  share it (nb7 behaviour);
- fixed seeds (42) and single-threaded CPU fits: XGBoost uses tree_method
  "hist" with n_jobs=1; LightGBM adds deterministic=True + force_row_wise=True;
  CatBoost pins thread_count=1 and disables its artifact writing. Two fits on
  the same data produce identical predictions — CPU-deterministic by design.

Dependencies are optional on other machines: nothing imports lightgbm or
catboost at module import time. check_dependencies() probes them so the
runner's factory can turn a missing dependency into a SKIPPED row
(fraud_pipeline.ArmSkipped) instead of a crash.

CV capability: the ensemble is 3 x 200 trees on ~4k rows — a few seconds per
fold — so it is cheap enough for full 5-fold cross-validation. It registers
with supports_cv=True; cross_validate_model clones it through the sklearn
get_params() contract (all constructor params are plain member-hyperparameter
dicts, members are built inside fit(), so cloning carries no fitted state).

Notebook API:
  from arms_gbdt import make_ensemble, make_member, member_params
  model = make_ensemble(y_fit)
  model.fit(X_fit, y_fit)
  proba = model.predict_proba(X_test)[:, 1]
"""

import numpy as np
from sklearn.base import BaseEstimator

RANDOM_SEED = 42

MEMBER_NAMES = ("xgb", "lgbm", "catboost")

# JSON special characters LightGBM bans in feature names ("Do not support
# special JSON characters in feature name") plus the quote/backslash pair.
_JSON_UNSAFE = str.maketrans({char: "_" for char in ",:[]{}\"'\\"})


def sanitize_feature_names(X):
    # LightGBM-safe copy of a feature matrix: values, row order and column
    # order unchanged, only names mapped. fraud_pipeline.encode_dummies strips
    # XGBoost's banned [ ] < but leaves the amount_eur_bucket interval labels'
    # comma, which LightGBM rejects. Pure per-name mapping — idempotent, so
    # fit() and predict_proba() always hand the members identical names.
    X = X.copy()
    X.columns = [str(column).translate(_JSON_UNSAFE) for column in X.columns]
    if X.columns.duplicated().any():
        dupes = X.columns[X.columns.duplicated()].tolist()
        raise ValueError("feature-name sanitation produced duplicate columns: %s" % dupes)
    return X


def check_dependencies():
    # Probe-import the three member libraries. Returns None when everything is
    # installed, otherwise a short message naming the missing module(s) — the
    # runner's factory turns a non-None result into an ArmSkipped row.
    missing = []
    for module in ("xgboost", "lightgbm", "catboost"):
        try:
            __import__(module)
        except ImportError as exc:
            missing.append("%s (%s)" % (module, exc))
    return "; ".join(missing) if missing else None


def member_params(y_fit):
    # Per-member constructor params, shared by every use of the arm (final
    # fit, validation carve, CV clones). y_fit drives the rare-class weighting:
    # the neg/pos ratio of the fitting carve, applied per member.
    y_arr = np.asarray(y_fit).ravel()
    scale_pos_weight = float((y_arr == 0).sum() / (y_arr == 1).sum())
    return {
        "xgb_params": dict(
            # Mirrors fraud_pipeline.FIXED_XGB_PARAMS + determinism pins.
            learning_rate=0.05,
            max_depth=6,
            n_estimators=200,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            tree_method="hist",  # fast, CPU-deterministic
            n_jobs=1,
            verbosity=0,
        ),
        "lgbm_params": dict(
            # reg_lambda keeps LightGBM's default 0.0 (its ridge term lives in
            # lambda_l2, left at the library default here).
            learning_rate=0.05,
            max_depth=6,
            n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            deterministic=True,   # CPU reproducibility
            force_row_wise=True,  # stable row-wise histogram path (also silences the hint)
            n_jobs=1,
            verbosity=-1,
        ),
        "catboost_params": dict(
            # l2_leaf_reg keeps CatBoost's default 3.0.
            learning_rate=0.05,
            depth=6,
            iterations=200,
            class_weights=[1.0, scale_pos_weight],
            random_seed=RANDOM_SEED,
            thread_count=1,
            allow_writing_files=False,  # no catboost_info/ artifacts
            verbose=False,
        ),
    }


def make_member(name, y_fit):
    # One ensemble member as a standalone sklearn-style estimator — used by the
    # nb9 notebook to run each member through the runner protocol on its own.
    from catboost import CatBoostClassifier  # lazy: optional dependency
    from lightgbm import LGBMClassifier  # lazy: optional dependency
    from xgboost import XGBClassifier  # lazy: optional dependency

    params = member_params(y_fit)
    if name == "xgb":
        return XGBClassifier(**params["xgb_params"])
    if name == "lgbm":
        return LGBMClassifier(**params["lgbm_params"])
    if name == "catboost":
        return CatBoostClassifier(**params["catboost_params"])
    raise ValueError("unknown ensemble member: %s (expected one of %s)" % (name, MEMBER_NAMES))


def make_ensemble(y_fit):
    # Registry factory (fraud_pipeline._make_gbdt_ensemble): the soft-vote
    # ensemble with per-member weights read from the fitting carve's y.
    return GBDTSoftVoteEnsemble(**member_params(y_fit))


class GBDTSoftVoteEnsemble(BaseEstimator):
    """Equal-weight soft vote of XGB + LightGBM + CatBoost on one feature matrix.

    sklearn-style wrapper the registry and cross_validate_model can clone:
    every constructor param is a plain dict of member hyperparameters
    (member_params) and fit() builds the members, so get_params()/clone
    reconstruct the estimator without carrying fitted state. predict_proba()
    returns the mean of the members' positive-class probabilities as an
    (n_samples, 2) array — column 1 is the ensemble fraud score.
    """

    def __init__(self, xgb_params=None, lgbm_params=None, catboost_params=None):
        self.xgb_params = xgb_params
        self.lgbm_params = lgbm_params
        self.catboost_params = catboost_params

    def fit(self, X, y):
        from catboost import CatBoostClassifier  # lazy: optional dependency
        from lightgbm import LGBMClassifier  # lazy: optional dependency
        from xgboost import XGBClassifier  # lazy: optional dependency

        X = sanitize_feature_names(X)
        y_arr = np.asarray(y).ravel()
        self.xgb_ = XGBClassifier(**dict(self.xgb_params or {})).fit(X, y_arr)
        self.lgbm_ = LGBMClassifier(**dict(self.lgbm_params or {})).fit(X, y_arr)
        self.catboost_ = CatBoostClassifier(**dict(self.catboost_params or {})).fit(
            X, y_arr
        )
        self.members_ = {
            "xgb": self.xgb_,
            "lgbm": self.lgbm_,
            "catboost": self.catboost_,
        }
        self.classes_ = np.array([0, 1])
        return self

    def member_proba(self, X):
        # Positive-class probability per member: {name: 1-d array}.
        X = sanitize_feature_names(X)
        return {
            name: np.asarray(model.predict_proba(X))[:, 1]
            for name, model in self.members_.items()
        }

    def predict_proba(self, X):
        p1 = np.mean(np.column_stack(list(self.member_proba(X).values())), axis=1)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
