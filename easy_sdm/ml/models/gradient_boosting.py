from sklearn.ensemble import GradientBoostingClassifier


class GradientBoosting(GradientBoostingClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.framework = "sklearn"
        self.estimator_name = "GradientBoosting"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs):
        super().fit(X_train, y_train, **kwargs)

    def predict(self, x):
        return super().predict(x)

    def predict_adaptability(self, x):
        return super().predict_proba(x)[:, 1].reshape(-1, 1)
