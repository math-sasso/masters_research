import mlflow
import numpy as np
import xgboost
from easy_sdm.configs import configs
from pathlib import Path
from easy_sdm.enums.estimator_type import EstimatorType
from easy_sdm.featuarizer import MinMaxScalerWrapper

from easy_sdm.typos.species import Species
from easy_sdm.utils.data_loader import (
    NumpyArrayLoader,
    RasterLoader,
    PickleLoader,
    DatasetLoader,
)
from .persistance.map_results_persistance import MapResultsPersistance


class Prediction_Job:

    """
        the output directory is prepared before logging:
        ├── output
        │   ├── data
        │   │   ├── data_sample.csv
        │   │   └── data_sample.html
        │   ├── images
        │   │   ├── gif_sample.gif
        │   │   └── image_sample.png
        │   ├── maps
        │   │   └── map_sample.geojson
        │   └── plots
        │       └── plot_sample.html
    """

    def __init__(self, data_dirpath: Path) -> None:
        self.configs = configs
        self.data_dirpath = data_dirpath
        self.__setup()
        self.__setup_mlflow()

    def __setup(self):

        land_reference = RasterLoader(
            self.data_dirpath / "raster_processing/region_mask.tif"
        ).load_dataset()

        raster_path_list = PickleLoader(
            self.data_dirpath / "environment/relevant_raster_list"
        ).load_dataset()
        statistics_dataset, _ = DatasetLoader(
            self.data_dirpath / "featuarizer/raster_statistics.csv"
        ).load_dataset()

        self.scaler = MinMaxScalerWrapper(
            raster_path_list=raster_path_list, statistics_dataset=statistics_dataset
        )

        self.idx = np.where(
            land_reference.read(1) == self.configs["mask"]["positive_mask_val"]
        )
        self.loaded_model = None

    def __setup_mlflow(self):
        ml_dirpath = str(Path.cwd() / "data/ml")
        mlflow.set_tracking_uri(f"file:{ml_dirpath}")

    def set_model(self, run_id: str, species: Species):

        # mlflow.get_run()
        logged_model = f"runs:/{run_id}/{species.get_name_for_plots()}"
        # self.loaded_model = mlflow.pyfunc.load_model(logged_model)
        history = mlflow.get_run(run_id).data.tags["mlflow.log-model.history"].lower()

        if "xgboost" in history:
            self.loaded_model = mlflow.xgboost.load_model(logged_model)
        elif "tabnet" in history:
            self.loaded_model = mlflow.pytorch.load_model(logged_model)
        else:
            self.loaded_model = mlflow.sklearn.load_model(logged_model)

        self.species = species
        self.run_id = run_id

    def map_prediction(self):

        assert self.loaded_model != None, "Set model first"
        stacked_raster_coverages = NumpyArrayLoader(
            self.data_dirpath / "environment/environment_stack.npy"
        ).load_dataset()

        coverages_of_interest = stacked_raster_coverages[:, self.idx[0], self.idx[1]].T
        scaled_coverages = self.scaler.scale_coverages(coverages_of_interest)
        global_pred = self.loaded_model.predict_adaptability(scaled_coverages)

        # num_env_vars = stacked_raster_coverages.shape[0]
        # global_pred = self.loaded_model.predict(stacked_raster_coverages.reshape(num_env_vars,-1).T)

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

    def log_map(self, Z: np.ndarray):
        map_persistance = MapResultsPersistance(
            species=self.species, data_dirpath=self.data_dirpath
        )
        output_path = map_persistance.create_result_adaptabilities_map(
            Z=Z, run_id=self.run_id
        )

        with mlflow.start_run(run_id=self.run_id) as run:
            mlflow.log_artifact(output_path)

    def dataset_prediction(self):
        pass
