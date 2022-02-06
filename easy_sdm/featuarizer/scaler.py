from calendar import c
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class MinMaxScalerWrapper:
    """ "
    Mix max scale dataframe and enverionment stack
    Columns should be passed because they will keed the enverionment mssvariables ordoer
    """

    def __init__(
        self, raster_path_list: List[str], statistics_dataset: pd.DataFrame
    ) -> None:
        self.raster_path_list = raster_path_list
        self.env_var_names = [
            Path(raster_path).name for raster_path in self.raster_path_list
        ]
        self.statistics_dataset = statistics_dataset.set_index("raster_name").reindex(
            self.env_var_names
        )

    def __scale(self, data: np.ndarray):
        min_vec = self.statistics_dataset["min"].to_numpy()
        max_vec = self.statistics_dataset["max"].to_numpy()
        data_std = (data - min_vec) / (max_vec - min_vec)
        data_scaled = data_std * (max_vec - min_vec) + min_vec
        return data_scaled

    def scale_df(self, df: pd.DataFrame):

        label = df["label"].to_numpy()
        df = df.drop("label", axis=1)
        columns = df.columns
        values = df.to_numpy()
        values_scaled = self.__scale(values)
        scaled_df = pd.DataFrame(values_scaled, columns=columns)
        scaled_df["label"] = label
        return scaled_df

    def scale_stack(self, stack):
        stack = stack.transpose(1, 2, 0)
        stack_scaled = self.__scale(stack)
        stack_scaled = stack_scaled.transpose(2, 0, 1)
        return stack_scaled
