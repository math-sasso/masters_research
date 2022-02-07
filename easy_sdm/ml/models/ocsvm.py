from sklearn import svm


class OCSVM:
    def __init__(self, **hyperparams) -> None:
        self.hyperparams = hyperparams

    def fit(self, x):
        self.clf = svm.OneClassSVM(**self.hyperparams)
        self.clf.fit(x)

    def predict(self, x):
        return self.clf.predict(x)

    def get_decision_function(self, x):
        return self.clf.decision_function(x)
