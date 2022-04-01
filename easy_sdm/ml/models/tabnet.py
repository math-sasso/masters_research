from ..metrics.auc import AUC
from ..metrics.tss import TSS
from ..metrics.kappa import Kappa
from pytorch_tabnet.tab_model import TabNetClassifier


class TabNetProxy(TabNetClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(self, X_train, y_train, X_valid, y_valid, **kwargs):

        super().fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=[Kappa, AUC, TSS],
            **kwargs
        )

    def predict_adaptability(self, X_test):
        return super().predict_proba(X_test)[:, 1]
