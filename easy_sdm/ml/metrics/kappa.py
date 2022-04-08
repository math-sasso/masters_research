from pytorch_tabnet.metrics import Metric
from sklearn.metrics import cohen_kappa_score


class Kappa(Metric):
    def __init__(self):
        self._name = "kappa"
        self._maximize = True

    def __call__(self, y_true, y_score):
        # kappa = cohen_kappa_score(y_true, (y_score[:, 1] >= 0.5).astype(int))
        kappa = cohen_kappa_score(y_true, (y_score >= 0.5).astype(int))
        return kappa
