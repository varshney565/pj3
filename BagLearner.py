import numpy as np

class BagLearner:
    def __init__(self, learner, kwargs=None, bags=20, boost=False, verbose=False):
        self.learner_class = learner
        self.kwargs = {} if kwargs is None else dict(kwargs)
        self.bags = int(bags)
        self.boost = bool(boost)
        self.verbose = bool(verbose)
        self.learners = []
        self._rng = np.random.default_rng()

    def author(self):
        return "vsingla31"

    def study_group(self):
        return "vsingla31"

    def add_evidence(self, data_x, data_y):
        X = np.asarray(data_x, dtype=float)
        y = np.asarray(data_y, dtype=float).reshape(-1)
        n = X.shape[0]
        self.learners = []
        for _ in range(self.bags):
            idx = self._rng.integers(0, n, size=n)
            model = self.learner_class(**self.kwargs)
            model.add_evidence(X[idx], y[idx])
            self.learners.append(model)

    def query(self, points):
        X = np.asarray(points, dtype=float)
        preds = np.zeros(X.shape[0], dtype=float)
        for m in self.learners:
            preds += m.query(X)
        return preds / float(len(self.learners)) if self.learners else preds
