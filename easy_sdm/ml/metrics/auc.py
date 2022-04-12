from .base import BaseMetric
from sklearn.metrics import roc_auc_score


class AUC(BaseMetric):
    def __init__(self):
        self._name = "auc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_score = self.__adjust_y_score(y_score=y_score)
        auc = roc_auc_score(y_true, y_score)
        return auc
