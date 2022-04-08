from pytorch_tabnet.metrics import Metric
from sklearn.metrics import confusion_matrix


class TSS(Metric):
    def __init__(self):
        self._name = "tss"
        self._maximize = True

    def __call__(self, y_true, y_score):
        # cm = confusion_matrix(y_true, (y_score[:, 1] >= 0.5).astype(int))
        cm = confusion_matrix(y_true, (y_score >= 0.5).astype(int))
        TP = cm[1][1]
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        sensitivity = TP / float(TP + FN)
        specificity = TN / float(TN + FP)
        TSS = sensitivity + specificity - 1
        return TSS
