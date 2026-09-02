"""F5 threshold/statistical scorers over the fused latent space (FOC-179).

The dictionary model's threshold logic (nb4) ported to the latent space: every
scorer here is an anomaly score over the fused 1280-d embedding (L2-normalized
face(512)+text(384)+demo(384) blocks, stateless concat — see arms_fusion), and
its threshold is calibrated on the fitting rows only, the same story the
dictionary model applies over its per-variable fraud probabilities
(train-only 3-threshold F1 grid in DictionaryRateEnricher).

Leakage discipline (the binding constraint, F4 lesson): every fitted object —
centroid, neighbor index, k-means, PCA, GMM, logistic weights — is fitted
inside fit() on the rows the runner hands the estimator (the stratified fit
carve), never on the full pre-split frame. Cluster/density scorers are fitted
on the LEGITIMATE fitting rows only: they model the legitimate terrain, and
the anomaly score is how far a row sits outside it.

Protocol contract (fraud_pipeline.run_arm_on_split): the runner freezes its
own best-F1 threshold on a validation carve and evaluates every arm through
the same path, so each scorer exposes its raw anomaly score as
predict_proba(X)[:, 1]; the arm's own train-percentile operating point is
stored (train_percentile_) for the nb16 calibration study and does NOT
interfere with the frozen-threshold comparison against the 13 existing arms.

Determinism: every randomized component gets an explicit fixed seed
(RANDOM_STATE = 42, the repo convention) — PCA randomized SVD, k-means ++,
GMM init — so a re-run reproduces byte-identical result rows (no wall-clock
fields, sort_keys JSONL, same as the F4 arms).
"""

import numpy as np
from sklearn.base import BaseEstimator

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Shared scaffolding for the train-calibrated anomaly scorers.
class _LatentAnomalyBase(BaseEstimator):
    """fit() calibrates on the fitting rows; predict_proba returns the score.

    Subclasses implement _fit_reference(X_ref) (fitted on legitimate fitting
    rows) and _score(X) (higher = more anomalous). The base class stores the
    legitimate-row percentile of the fitting scores as train_percentile_ —
    the latent-space analogue of the dictionary model's train-only threshold
    grid — and exposes predict() as the flag at that operating point.
    """

    def __init__(self, percentile=99.0):
        self.percentile = percentile

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y).ravel().astype(int)
        legit = X_arr[y_arr == 0]
        if len(legit) < 2:
            raise ValueError("need at least 2 legitimate fitting rows")
        self._fit_reference(legit)
        # Percentile over the LEGITIMATE fitting scores: the reference terrain
        # defines "normal", so the operating point is its own tail. Scorers
        # whose reference contains the query rows override _calibration_scores
        # to drop their self-match by row identity.
        self.train_percentile_ = float(
            np.percentile(self._calibration_scores(legit), self.percentile)
        )
        self.reference_rows_ = int(len(legit))
        return self

    def _fit_reference(self, X_legit):  # pragma: no cover - interface
        raise NotImplementedError

    def _score(self, X):  # pragma: no cover - interface
        raise NotImplementedError

    def _calibration_scores(self, X_legit):
        # Scores of the reference rows themselves, as consumed by the
        # percentile above. Default: identical to _score (centroid/density
        # scorers have no per-row self-entry to remove).
        return self._score(X_legit)

    def predict_proba(self, X):
        scores = self._score(np.asarray(X, dtype=np.float64))
        # Column 1 is what the runner reads: the raw anomaly score. Column 0
        # exists only for sklearn predict() symmetry; the scores are distances
        # / densities, not probabilities — the runner's frozen threshold and
        # AUCs are rank-based, so the raw score is the comparable quantity.
        return np.column_stack([-scores, scores])

    def predict(self, X):
        return (self._score(np.asarray(X, dtype=np.float64)) >= self.train_percentile_).astype(int)


class NearestLegitScorer(_LatentAnomalyBase):
    """Distance to the k-th nearest legitimate fitting row.

    Local-density version of the dictionary story: a transaction whose fused
    embedding has no close legitimate neighbour is anomalous. When the
    legitimate fitting rows are scored for calibration, each row's own
    reference entry is excluded BY ROW IDENTITY, not by a distance cutoff —
    sklearn's self distance is ~1e-8 of numeric noise rather than exact 0, so
    a cutoff both misses the self-match and could misfire on genuine
    near-duplicates.
    """

    def __init__(self, n_neighbors=5, percentile=99.0, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.percentile = percentile
        self.metric = metric

    def _fit_reference(self, X_legit):
        from sklearn.neighbors import NearestNeighbors  # lazy: keeps import light

        self.n_neighbors_ = int(min(self.n_neighbors, len(X_legit) - 1))
        self.nn_ = NearestNeighbors(
            n_neighbors=self.n_neighbors_, metric=self.metric
        ).fit(X_legit)

    def _score(self, X):
        # Out-of-sample rows: no self-match exists in the reference, so the
        # k-th neighbour is genuine. Reference rows go through
        # _calibration_scores, which removes their self-match by identity.
        dists, _ = self.nn_.kneighbors(X, n_neighbors=self.n_neighbors_)
        return dists[:, -1]

    def _calibration_scores(self, X_legit):
        # Self-exclusion by ROW IDENTITY (review fix): kneighbors returns
        # positions into the reference and query i IS reference row i, so the
        # self-match is masked by index. A distance cutoff cannot do this —
        # sklearn's self distance is ~1e-8 of dot-product-expansion noise,
        # not exact 0, which slipped past the old < 1e-12 heuristic and let
        # the row's own match deflate the percentile calibration.
        dists, idxs = self.nn_.kneighbors(X_legit, n_neighbors=self.n_neighbors_ + 1)
        is_self = idxs == np.arange(len(X_legit))[:, None]
        masked = np.where(is_self, np.inf, dists)
        masked.sort(axis=1)
        # k+1 columns: with self present k genuine neighbours remain; if an
        # exact duplicate won the tie and pushed self out, k+1 genuine remain.
        # The k-th genuine neighbour sits at position k-1 of the sorted row
        # in both cases.
        return masked[:, self.n_neighbors_ - 1]


class CentroidScorer(_LatentAnomalyBase):
    """Euclidean distance to the legitimate fitting centroid."""

    def _fit_reference(self, X_legit):
        self.centroid_ = X_legit.mean(axis=0)

    def _score(self, X):
        return np.linalg.norm(X - self.centroid_, axis=1)


class CosineCentroidScorer(_LatentAnomalyBase):
    """Angular score: 1 - cosine to the legitimate mean direction.

    The fused blocks are L2-normalized per modality, so the informative
    geometry is the angle, not the radius — the dictionary model's angle
    threshold counterpart.
    """

    def _fit_reference(self, X_legit):
        norms = np.linalg.norm(X_legit, axis=1, keepdims=True)
        unit = X_legit / np.clip(norms, 1e-12, None)
        centroid = unit.mean(axis=0)
        self.ref_direction_ = centroid / max(np.linalg.norm(centroid), 1e-12)

    def _score(self, X):
        unit = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        return 1.0 - unit @ self.ref_direction_


class ClusterAnomalyScorer(_LatentAnomalyBase):
    """Within-cluster anomaly: k-means on the LEGITIMATE fitting rows only.

    The plan's "anomaly within clusters" method with the leakage discipline
    stated in FOC-179: clusters are fitted on train (here: the legitimate fit
    carve) and test rows are scored by the distance to the nearest cluster
    center — no k-means ever sees the full dataset or the labels.
    """

    def __init__(self, n_clusters=8, percentile=99.0, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.percentile = percentile
        self.random_state = random_state

    def _fit_reference(self, X_legit):
        from sklearn.cluster import KMeans  # lazy

        k = int(min(self.n_clusters, len(X_legit)))
        self.kmeans_ = KMeans(
            n_clusters=k, n_init=10, random_state=self.random_state
        ).fit(X_legit)

    def _score(self, X):
        return np.linalg.norm(
            X[:, None, :] - self.kmeans_.cluster_centers_[None, :, :], axis=2
        ).min(axis=1)


class GMMDensityScorer(_LatentAnomalyBase):
    """Distributional threshold: GMM density over the legitimate terrain.

    The closest latent-space analogue of the dictionary model's distributional
    thresholds over per-variable fraud probabilities: model the legitimate
    score distribution explicitly. PCA (fitted on legitimate fitting rows
    ONLY) reduces 1280-d to n_pca components, then a diagonal-covariance GMM
    estimates the legitimate density; the anomaly score is the negative log
    likelihood. Both fits live inside fit() — train-only per split.
    """

    def __init__(
        self,
        n_pca=64,
        n_components=4,
        percentile=99.0,
        random_state=RANDOM_STATE,
    ):
        self.n_pca = n_pca
        self.n_components = n_components
        self.percentile = percentile
        self.random_state = random_state

    def _fit_reference(self, X_legit):
        from sklearn.decomposition import PCA  # lazy
        from sklearn.mixture import GaussianMixture  # lazy

        self.pca_ = PCA(
            n_components=int(min(self.n_pca, len(X_legit) - 1, X_legit.shape[1])),
            random_state=self.random_state,
        ).fit(X_legit)
        legit_pcs = self.pca_.transform(X_legit)
        self.gmm_ = GaussianMixture(
            n_components=int(min(self.n_components, max(2, len(legit_pcs) // 50))),
            covariance_type="diag",
            random_state=self.random_state,
        ).fit(legit_pcs)

    def _score(self, X):
        return -self.gmm_.score_samples(self.pca_.transform(X))


class LatentLogisticClassifier(BaseEstimator):
    """Mateusz's F5 decision: one light classifier arm over the latent space.

    Plain L2 logistic regression on the fused 1280-d embedding — no fitted
    dimensionality reduction (the runner protocol already fits per split, and
    LR is cheap at n~3k x d~1.3k); deterministic lbfgs solver. Included in the
    same registry so it is evaluated through the identical protocol as the
    threshold scorers and the 13 existing arms.
    """

    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression  # lazy

        self.model_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs"
        )
        self.model_.fit(np.asarray(X, dtype=np.float64), np.asarray(y).ravel())
        self.coef_norm_ = float(np.linalg.norm(self.model_.coef_))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float64))

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=np.float64))
