from pytorch_tabnet.metrics import Metric

class BaseMetric(Metric):

    def __adjust_y_score(self, y_score):
        """Get only the positive probabilities

        Args:
            y_score (numpy_array): Can be 1D or 2D.  Must be the predict proba of positive cases.
        """
        if y_score.shape[1] == 2:
            y_score=y_score[:,1]
        elif y_score.shape[1] == 1:
            pass #it is ok
        else:
            ValueError("y_score shape cant be zero")

    def __call__(self, y_true, y_score):
        NotImplementedError()