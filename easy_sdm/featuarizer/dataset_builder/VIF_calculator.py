from cmath import inf
from pathlib import Path
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from easy_sdm.utils import DatasetLoader


class VIFCalculator:
    def __init__(self) -> None:
        self.X = None

    def calculate_vif(self, X: pd.DataFrame):

        max = inf
        while max > 10:
            # VIF dataframe
            vif_data = pd.DataFrame()
            vif_data["feature"] = X.columns

            # calculating VIF for each feature
            vif_data["VIF"] = [
                variance_inflation_factor(X.values, i) for i in range(len(X.columns))
            ]

            idmax = vif_data["VIF"].idxmax()
            max = vif_data["VIF"].max()
            feature_to_remove = vif_data.iloc[idmax]["feature"]
            X = X.drop([feature_to_remove], axis=1)

        self.X = X
        self.vif_data = vif_data

    def get_optimous_columns(self):
        assert self.X is not None
        return self.X.columns

    def get_vif_df(self):
        assert self.vif_data is not None
        return self.vif_data

    def get_optimouns_df(self):
        assert self.X is not None
        return self.X


# if __name__ == "__main__":
#     datataset_loader = DatasetLoader(dataset_path=Path.cwd() / "data/featuarizer/dataset.csv",
#               output_column=  "label")

#     X, y = datataset_loader.load_dataset()
#     vif_calculator = VIFCalculator()
#     vif_calculator.calculate_vif(X)
