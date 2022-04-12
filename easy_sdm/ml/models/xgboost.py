from xgboost import XGBClassifier, XGBRFClassifier

class Xgboost(XGBClassifier):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        self.framework = "xgboost"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs):
        super().fit(X_train.to_numpy(), y_train.to_numpy().ravel(), **kwargs)
        import pdb;pdb.set_trace()
    def predict(self, x):
        return super().predict(x)

    def predict_adaptability(self, x):
        return super().predict_proba(x)[:, 1]

class XgboostRF(XGBRFClassifier):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        self.framework = "xgboost"

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs):
        super().fit(X_train, y_train, **kwargs)

    def predict(self, x):
        return super().predict(x)

    def predict_adaptability(self, x):
        return super().predict_proba(x)[:, 1]