import mlflow
from typing import Dict, Optional
from pathlib import Path


class MLFlowPersistence:
    def __init__(
        self,
        mlflow_uri: str,
        mlflow_experiment_name: str,
        mlflow_databricks_dirpath: Optional[str],
    ) -> None:
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_databricks_dirpath = mlflow_databricks_dirpath
        self.__set_experiment()

    def __set_experiment(self):

        if self.mlflow_uri:
            print(f"self.mlflow_uri: {self.mlflow_uri}")
            mlflow.set_tracking_uri(self.mlflow_uri)
            tracked_uri = mlflow.get_tracking_uri()
            print(f"Tracked URI: {tracked_uri}")
            mlflow.set_experiment(self.mlflow_experiment_name)
        else:
            print("MLFLOW_URI is not set")

    def __persist_logs(self, metrics: Dict, parameters: Dict):
        print(f"Logged Parameters {parameters}")
        print(f"Logged Metrics {metrics}")
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

    def persist(self, model, metrics: Dict, parameters: Dict, end=True):

        if mlflow.active_run():
            mlflow.end_run()

        run = mlflow.start_run()
        run_id = run.info.run_id

        mlflow.set_tag("run ID", run_id)

        self.__persist_logs(metrics, parameters)

        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        if end:
            mlflow.end_run()

    def persist_and_register(self, model_name, model, metrics: Dict, parameters: Dict):
        """model_name: repository name"""
        self.persist(model, metrics, parameters, end=False)
        run_uuid = mlflow.active_run().info.run_uuid
        mlflow.register_model("runs:/{}/{}".format(run_uuid, "model"), model_name)
        mlflow.end_run()
