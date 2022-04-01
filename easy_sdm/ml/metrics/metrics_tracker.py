from auc import AUC
from kappa import Kappa
from tss import TSS


class MetricsTracker:
    def __init__(self,) -> None:

        self.metrics = {}

    def get_metrics(self, y_score, y_true):

        self.metrics["auc"] = AUC()(y_score=y_score, y_true=y_true)
        self.metrics["kappa"] = AUC()(y_score=y_score, y_true=y_true)
        self.metrics["tss"] = AUC()(y_score=y_score, y_true=y_true)

        return self.metrics
