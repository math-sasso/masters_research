import numpy as np
from easy_sdm.configs import configs
from pathlib import Path

class Prediction_Job():
    def __init__(self) -> None:
        self.configs = configs
        self.environment_dirpath = Path.cwd() / "data/environment"

    # def setup(self):
    #     pass

    # def map_prediction(self):

    #     Z= np.ones((stacked_raster_coverages_shape[1], stacked_raster_coverages_shape[2]), dtype=np.float32)
    #     # Z *= global_pred.min()
    #     # Z *=-1 #This will be necessary to set points outside map to the minimum
    #     Z*= self.configs["maps"]["no_data_val"] #This will be necessary to set points outside map to the minimum
    #     Z[idx[0], idx[1]] = global_pred.ravel()
    #     Z[Z == raster_utils.no_data_val] = -0.001
    #     return Z

    # def dataset_prediction(self):
    #     pass