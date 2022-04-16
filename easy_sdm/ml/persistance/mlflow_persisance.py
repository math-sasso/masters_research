import mlflow
from typing import Dict
from pathlib import Path


class MLFlowPersistence:
    def __init__(self, mlflow_experiment_name: str,) -> None:
        self.mlflow_experiment_name = mlflow_experiment_name
        self.__setup_mlflow()
        self.__set_experiment()

    def __setup_mlflow(self):
        ml_dirpath = str(Path.cwd() / "data/ml")
        mlflow.set_tracking_uri(f"file:{ml_dirpath}")

    def __set_experiment(self):

        if not mlflow.get_experiment_by_name(self.mlflow_experiment_name):
            mlflow.create_experiment(name=self.mlflow_experiment_name)

    def __persist_logs(self, metrics: Dict, parameters: Dict):
        print(f"Logged Parameters {parameters}")
        print(f"Logged Metrics {metrics}")
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        # mlflow.log_artifact('roi_features.csv', artifact_path='features')

    def persist(self, model, metrics: Dict, parameters: Dict, vif: str, end=True):

        if mlflow.active_run():
            mlflow.end_run()

        run = mlflow.start_run()
        run_id = run.info.run_id

        mlflow.set_tag("VIF", vif)
        mlflow.set_tag("run ID", run_id)

        self.__persist_logs(metrics, parameters)

        if model.framework == "sklearn":
            mlflow.sklearn.log_model(model, self.mlflow_experiment_name)
        elif model.framework == "pytorch":
            mlflow.pytorch.log_model(model, self.mlflow_experiment_name)
        elif model.framework == "xgboost":
            mlflow.xgboost.log_model(model, self.mlflow_experiment_name)
        else:
            raise TypeError()

        if end:
            mlflow.end_run()

    def persist_and_register(self, model_name, model, metrics: Dict, parameters: Dict):
        """model_name: repository name"""
        self.persist(model, metrics, parameters, end=False)
        run_uuid = mlflow.active_run().info.run_uuid
        mlflow.register_model("runs:/{}/{}".format(run_uuid, "model"), model_name)
        mlflow.end_run()
