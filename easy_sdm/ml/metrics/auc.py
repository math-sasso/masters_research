from pytorch_tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score


class AUC(Metric):
    def __init__(self):
        self._name = "auc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        auc = roc_auc_score(y_true, y_score[:, 1])
        return auc
