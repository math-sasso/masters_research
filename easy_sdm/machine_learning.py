@dataclass
class TrainJob(BaseTrainJob):
    train_data_loader: BaseDataLoader
    validation_data_loader: BaseDataLoader
    estimator: BaseEstimator
    estimator_parameters: Dict
    output_path: Path
    mlflow_uri: str
    mlflow_experiment_name: str
    mlflow_databricks_dirpath: Optional[str]

    def setup(self):
        logger.info("TrainJob.setup begin...")

        # SETUP MLFLOW
        self.mlflow_persister = MLFlowPersistence(
            self.mlflow_uri, self.mlflow_experiment_name, self.mlflow_databricks_dirpath
        )

        # LOAD DATASETS
        (
            self.train_features,
            self.y_train_df,
            self.ids_train_df,
        ) = self.train_data_loader.load_dataset()
        (
            self.prediction_features,
            self.y_true_df,
            self.pred_ids_df,
        ) = self.validation_data_loader.load_dataset()

        # SETTING PIPELINE
        dict_dtypes = dict(self.train_features.dtypes)
        numerical_features = [k for k, v in dict_dtypes.items() if v in [float, int]]
        categorical_features = [
            x for x in self.train_features.columns if x not in numerical_features
        ]
        self.pipeline = SklearnPipeline(
            model=self.estimator,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        # SETTING METRICS TRACKER
        self.metrics_tracker = MetricsTracker()

    def fit(self):
        logger.info("TrainJob.fit begin...")

        self.pipeline.fit(self.train_features, self.y_train_df)
        self.__validate()

    def __validate(self):
        logger.info("TrainJob.valid begin...")

        self.predictions = self.pipeline.predict(X_hat=self.prediction_features)

        self.metrics = self.metrics_tracker.get_metrics(
            y_true=self.y_true_df, y_hat=self.predictions
        )

    def persist(self):
        EstimatorPersistence.dump(estimator=self.pipeline, output_dir=self.output_path)
        DataPersistence.features_payload_profiling(
            value=self.train_features, output_dir=self.output_path
        )
        ExperimentPersistence.save_scores(self.metrics, self.output_path, "metrics")
        ExperimentPersistence.save_predictions(
            self.predictions, self.output_path, "predictions"
        )
        self.mlflow_persister.persist(
            model=self.pipeline,
            metrics=self.metrics,
            parameters=self.estimator_parameters,
        )
