from enum import Enum


class EstimatorType(Enum):
    Tabnet = "Tabnet"
    MLP = "MLP"
    EnsembleForest = "EnsembleForest"
    GradientBoosting = "GradientBoosting"
    Xgboost = "Xgboost"
    XgboostRF = "XgboostRF"
    OCSVM = "OCSVM"
    Autoencoder = "Autoencoder"
