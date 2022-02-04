##########

import os
import numpy as np
import geopandas as gpd
from typing import List, Tuple, Dict
from sklearn import svm, metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
import rasterio
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np


class BaseModel:
    def __init__(self, **hyperparams) -> None:
        self.hyperparams = hyperparams

    def fit(self, x, y):
        """ Fitting data with normalized data """
        pass

    def predict(self, x):
        """ """
