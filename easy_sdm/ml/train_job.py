from pathlib import Path

from easy_sdm.enums import EstimatorType
from easy_sdm.ml.models import ocsvm
from easy_sdm.typos import Species
from easy_sdm.utils import DatasetLoader
from configs import configs

from .metrics import MetricsTracker
from .models import (
    MLP,
    OCSVM,
    GradientBoosting,
    RandomForest,
    TabNet,
    Xgboost,
    XgboostRF,
)
from .persistance.mlflow_persisance import MLFlowPersistence


class TrainJob:
    def __init__(
        self,
        train_data_loader: DatasetLoader,
        validation_data_loader: DatasetLoader,
        estimator_type: EstimatorType,
        species: Species,
        output_path: Path = None,
    ) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.estimator_type = estimator_type
        self.species = species
        self.output_path = output_path

        estimator_selector = EstimatorSelector(self.estimator_type)
        estimator_selector.select_estimator()
        self.estimator = estimator_selector.get_estimator()
        self.estimator_parameters = estimator_selector.get_estimator_parameters()
        self.vif_columns = False
        self.__setup()

    def __build_empty_folders(self):
        raise NotImplementedError()

    def normal_setup(
        self, train_data_loader: DatasetLoader, validation_data_loader: DatasetLoader
    ):
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.__setup()

    def vif_setup(
        self,
        vif_train_data_loader: DatasetLoader,
        vif_validation_data_loader: DatasetLoader,
    ):
        self.vif_columns = True
        self.train_data_loader = vif_train_data_loader
        self.validation_data_loader = vif_validation_data_loader
        self.__setup()

    def __setup(self):

        # Path with data used in the experiment
        experiment_featurizer_path = "/".join(
            str(self.train_data_loader.dataset_path).split("/")[:-1]
        )

        # SETUP MLFLOW
        self.mlflow_experiment_name = self.species.get_name_for_plots()
        self.columns_considered = "vif_columns" if self.vif_columns else "all_columns"

        self.mlflow_persister = MLFlowPersistence(
            self.mlflow_experiment_name, experiment_featurizer_path
        )

        # LOAD DATASETS
        (self.X_train_df, self.y_train_df,) = self.train_data_loader.load_dataset()
        (self.x_valid_df, self.y_valid_df,) = self.validation_data_loader.load_dataset()

        # SETTING PIPELINE
        self.pipeline = self.estimator
        # dict_dtypes = dict(self.X_train_df.dtypes)
        # numerical_features = [k for k, v in dict_dtypes.items() if v in [float, int]]
        # categorical_features = [
        #     x for x in self.X_train_df.columns if x not in numerical_features
        # ]
        # self.pipeline = SklearnPipeline(
        #     model=self.estimator,
        #     numerical_features=numerical_features,
        #     categorical_features=categorical_features,
        # )

        # SETTING METRICS TRACKER
        self.metrics_tracker = MetricsTracker()

    def fit(self):
        self.pipeline.fit(
            self.X_train_df, self.y_train_df, self.x_valid_df, self.y_valid_df
        )
        self.__validate()

    def __validate(self):
        self.prediction_scores = self.pipeline.predict_adaptability(x=self.x_valid_df)
        self.metrics = self.metrics_tracker.get_metrics(
            y_true=self.y_valid_df, y_score=self.prediction_scores
        )

    def persist(self):
        # EstimatorPersistence.dump(estimator=self.pipeline, output_dir=self.output_path)
        # DataPersistence.features_payload_profiling(
        #     value=self.X_train_df, output_dir=self.output_path
        # )
        # ExperimentPersistence.save_scores(self.metrics, self.output_path, "metrics")
        # ExperimentPersistence.save_predictions(
        #     self.predictions, self.output_path, "predictions"
        # )
        self.mlflow_persister.persist(
            model=self.pipeline,
            metrics=self.metrics,
            parameters=self.estimator_parameters,
            vif=self.columns_considered,
        )


class EstimatorSelector:
    def __init__(self, estimator_type: EstimatorType) -> None:
        self.random_state = 1
        self.estimator_type = estimator_type

    def select_estimator(self):

        if self.estimator_type == EstimatorType.MLP:
            estimator = MLP(
                hidden_layer_sizes=(200, 100, 50, 20, 10),
                random_state=self.random_state,
                max_iter=8000,
            )
        elif self.estimator_type == EstimatorType.GradientBoosting:
            estimator = GradientBoosting(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                criterion="squared_error",
            )
        elif self.estimator_type == EstimatorType.RandomForest:
            estimator = RandomForest(max_depth=10, random_state=self.random_state)
        elif self.estimator_type == EstimatorType.Tabnet:
            estimator = TabNet(device_name="cpu")
        elif self.estimator_type == EstimatorType.Xgboost:
            estimator = Xgboost(use_label_encoder=False)
        elif self.estimator_type == EstimatorType.XgboostRF:
            estimator = XgboostRF(use_label_encoder=False)
        elif self.estimator_type == EstimatorType.OCSVM:
            estimator = OCSVM(
                nu=configs["OCSVM"]["nu"],
                kernel=configs["OCSVM"]["kernel"],
                gamma=configs["OCSVM"]["gamma"],
            )
        else:
            raise ValueError("Use one of the possible estimators")

        self.estimator = estimator

    def get_estimator(self):
        return self.estimator

    def get_estimator_parameters(self):
        params = self.estimator.__dict__
        params = {k: v for k, v in params.items() if len(str(v)) <= 250}
        return params
