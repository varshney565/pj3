import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner:
    def __init__(self, verbose=False):
        self.verbose = bool(verbose)
        self._bags = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False) for _ in range(20)]
    def author(self): return "your_gt_username"
    def add_evidence(self, data_x, data_y):
        for b in self._bags: b.add_evidence(data_x, data_y)
    def query(self, points):
        s = None
        for b in self._bags:
            p = b.query(points)
            s = p if s is None else s + p
        return s / float(len(self._bags))
