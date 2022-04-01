from sklearn.neural_network import MLPClassifier

class MLPClassifierProxy(MLPClassifier):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)

    def fit(X_train, y_train, X_valid=None, y_valid=None):
        super().fit(X_train, y_train)
        result = None
        if X_valid is not None and y_valid is not None:
            result 

    def predict(self, x):
        return self.clf.predict(x)

    def get_decision_function(self, x):
        return self.clf.decision_function(x)
