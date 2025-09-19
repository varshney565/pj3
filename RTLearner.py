import numpy as np

class RTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = int(leaf_size)
        self.verbose = bool(verbose)
        self.tree = None
        self._rng = np.random.default_rng()

    def author(self):
        return "your_gt_username"

    def study_group(self):
        return "your_gt_username"

    def _leaf(self, y):
        return np.array([[-1.0, float(np.mean(y)), np.nan, np.nan]], dtype=float)

    def _pick_random_feature(self, X):
        var = np.var(X, axis=0)
        valid = np.where(var > 0.0)[0]
        if valid.size == 0:
            return -1
        return int(self._rng.integers(0, valid.size).item() if hasattr(self._rng.integers(0,1),"item") else self._rng.integers(0, valid.size))

    def _build(self, X, y):
        n = X.shape[0]
        if n <= self.leaf_size or np.all(y == y[0]):
            return self._leaf(y)
        feat = self._pick_random_feature(X)
        if feat == -1:
            return self._leaf(y)
        split_val = np.median(X[:, feat])
        left = X[:, feat] <= split_val
        if not left.any() or left.all():
            return self._leaf(y)
        L = self._build(X[left], y[left])
        R = self._build(X[~left], y[~left])
        root = np.array([[float(feat), float(split_val), 1.0, float(L.shape[0] + 1)]], dtype=float)
        return np.vstack((root, L, R))

    def add_evidence(self, data_x, data_y):
        X = np.asarray(data_x, dtype=float)
        y = np.asarray(data_y, dtype=float).reshape(-1)
        self.tree = self._build(X, y)

    def _query_point(self, x):
        i = 0
        while True:
            feat = int(self.tree[i, 0])
            if feat == -1:
                return self.tree[i, 1]
            split_val = self.tree[i, 1]
            i += int(self.tree[i, 2] if x[feat] <= split_val else self.tree[i, 3])

    def query(self, points):
        X = np.asarray(points, dtype=float)
        return np.array([self._query_point(row) for row in X], dtype=float)
