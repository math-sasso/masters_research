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


class RelevantRastersSelector:
    def __init__(self, data_dirpath, run_id) -> None:
        self.data_dirpath = data_dirpath
        self.run_id = run_id
        self.__setup_mlflow()
        self.__setup()

    def __setup_mlflow(self):
        ml_dirpath = str(Path.cwd() / "data/ml")
        mlflow.set_tracking_uri(f"file:{ml_dirpath}")

    def __setup(self):
        relevant_raster_list = PickleLoader(
            self.data_dirpath / "environment/relevant_raster_list"
        ).load_dataset()
        experiment_dataset_path = Path(
            mlflow.get_run(self.run_id).data.tags["experiment_dataset_path"]
        )
        vif_decision_df, _ = DatasetLoader(
            experiment_dataset_path / "vif_decision_df.csv"
        ).load_dataset()
        self.vif_decision_columns = vif_decision_df["feature"].tolist()
        relevant_raster_name_list = [
            str(path).split("/")[-1].replace(".tif", "")
            for path in relevant_raster_list
        ]
        self.vif_relevant_raster_list_pos = [
            relevant_raster_name_list.index(elem) for elem in self.vif_decision_columns
        ]

    def get_vif_decision_columns(self):
        return self.vif_decision_columns

    def get_vif_relevant_raster_list_pos(self):
        return self.vif_relevant_raster_list_pos


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

    def __init__(self, data_dirpath: Path, run_id: str, species: Species) -> None:
        self.configs = configs
        self.data_dirpath = data_dirpath
        self.species = species
        self.run_id = run_id
        self.__setup_mlflow()
        self.__setup()

    def __setup(self):

        land_reference = RasterLoader(
            self.data_dirpath / "raster_processing/region_mask.tif"
        ).load_dataset()

        statistics_dataset, _ = DatasetLoader(
            self.data_dirpath / "featuarizer/raster_statistics.csv"
        ).load_dataset()

        if mlflow.get_run(self.run_id).data.tags["VIF"] == "vif_columns":
            self.relevant_raster_selector = RelevantRastersSelector(
                data_dirpath=self.data_dirpath, run_id=self.run_id
            )
            statistics_dataset = statistics_dataset[
                statistics_dataset["raster_name"].isin(
                    self.relevant_raster_selector.get_vif_decision_columns()
                )
            ]

        self.scaler = MinMaxScalerWrapper(statistics_dataset=statistics_dataset)

        self.idx = np.where(
            land_reference.read(1) == self.configs["mask"]["positive_mask_val"]
        )
        self.loaded_model = None

    def __setup_mlflow(self):
        ml_dirpath = str(Path.cwd() / "data/ml")
        mlflow.set_tracking_uri(f"file:{ml_dirpath}")

    def set_model(self):

        # mlflow.get_run()
        logged_model = f"runs:/{self.run_id}/{self.species.get_name_for_plots()}"
        # self.loaded_model = mlflow.pyfunc.load_model(logged_model)
        history = (
            mlflow.get_run(self.run_id).data.tags["mlflow.log-model.history"].lower()
        )

        if "xgboost" in history:
            self.loaded_model = mlflow.xgboost.load_model(logged_model)
        elif "tabnet" in history:
            self.loaded_model = mlflow.pytorch.load_model(logged_model)
        else:
            self.loaded_model = mlflow.sklearn.load_model(logged_model)

    def __get_stacked_raster_coverages(self):

        stacked_raster_coverages = NumpyArrayLoader(
            self.data_dirpath / "environment/environment_stack.npy"
        ).load_dataset()

        if mlflow.get_run(self.run_id).data.tags["VIF"] == "vif_columns":
            stacked_raster_coverages = stacked_raster_coverages[
                self.relevant_raster_selector.get_vif_relevant_raster_list_pos()
            ]

        return stacked_raster_coverages

    def map_prediction(self):

        assert self.loaded_model != None, "Set model first"

        stacked_raster_coverages = self.__get_stacked_raster_coverages()
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
