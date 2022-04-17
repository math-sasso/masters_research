import mlflow
import numpy as np
from easy_sdm.configs import configs
from pathlib import Path

from easy_sdm.typos.species import Species
from easy_sdm.utils.data_loader import NumpyArrayLoader, RasterLoader


class Prediction_Job:
    def __init__(self, data_dirpath: Path) -> None:
        self.configs = configs
        self.data_dirpath = data_dirpath
        self.__setup()
        self.__setup_mlflow()

    def __setup(self):

        land_reference = RasterLoader(
            self.data_dirpath
            / "raster_processing/region_mask.tif"
        ).load_dataset()

        self.idx = np.where(land_reference.read(1) == self.configs["mask"]["positive_mask_val"])
        self.loaded_model = None

    def __setup_mlflow(self):
        ml_dirpath = str(Path.cwd() / "data/ml")
        mlflow.set_tracking_uri(f"file:{ml_dirpath}")

    def set_model(self, run_id: str, species: Species):

        logged_model = f"runs:/{run_id}/{species.get_name_for_plots()}"
        self.loaded_model = mlflow.pyfunc.load_model(logged_model)

    def map_prediction(self):

        assert self.loaded_model != None, "Set model first"
        stacked_raster_coverages = NumpyArrayLoader(
            self.data_dirpath / "environment/environment_stack.npy"
        ).load_dataset()

        coverages_of_interest = stacked_raster_coverages[:, self.idx[0], self.idx[1]].T
        global_pred = self.loaded_model.predict(coverages_of_interest)
        #num_env_vars = stacked_raster_coverages.shape[0]
        #global_pred = self.loaded_model.predict(stacked_raster_coverages.reshape(num_env_vars,-1).T)

        Z = np.ones(
            (stacked_raster_coverages.shape[1], stacked_raster_coverages.shape[2]),
            dtype=np.float32,
        )
        # Z *= global_pred.min()
        # Z *=-1 #This will be necessary to set points outside map to the minimum
        Z *= self.configs["maps"][
            "no_data_val"
        ]  # This will be necessary to set points outside map to the minimum
        Z[self.idx[0], self.idx[1]] = global_pred.ravel()
        Z[Z == self.configs["maps"]["no_data_val"]] = -0.001
        return Z

    def dataset_prediction(self):
        pass
