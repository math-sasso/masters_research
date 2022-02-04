from abc import ABC
from argparse import ArgumentError
from typing import Dict

import numpy as np
import pandas as pd
import rasterio
from configs import configs

from easy_sdm.models import OCSVM


class BasePseudoSpeciesGenerator(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def fit(self, occurrence_data: pd.DataFrame):
        pass

    @abstractmethod
    def generate(self, poinst_to_generate: int):
        pass


class RandomPseudoSpeciesGenerator(BasePseudoSpeciesGenerator):
    def __init__(self) -> None:
        raise NotImplementedError("This class is not yet implemented")

    def generate(self):
        pass


class RSEPPseudoSpeciesGenerator(BasePseudoSpeciesGenerator):
    """[Generate points that are inside the proposed territory and in regions where data is an an anomaly]

    Args:
        BasePseudoSpeciesGenerator ([type]): [description]
    """

    def __init__(self, **kwargs) -> None:
        if any(kwargs.values) is None:
            raise ArgumentError("All expected  kwargs must be provided")

        self.hyperparameters = kwargs.get("hyperparameters", None)
        self.region_mask_raster = kwargs.get("region_mask_raster", None)
        self.stacked_raster_coverages = kwargs.get("stacked_raster_coverages", None)
        self.__check_arguments()
        self.configs = configs
        self.inside_mask_idx = np.where(
            self.region_mask_raster.read(1) != configs["mask"]["negative_mask_val"]
        )
        self.ocsvm = OCSVM(self.hyperparameters)
        self.pseudo_absense_label = 0

    def __check_arguments(self):
        if self.hyperparameters is not Dict:
            raise ArgumentError()
        if self.region_mask_raster is not rasterio.io.DatasetReader:
            raise ArgumentError()
        if self.stacked_raster_coverages is not np.ndarray:
            raise ArgumentError()

    def fit(self, occurrence_df: pd.DataFrame):
        # Coords X and Y in two tuples where condition matchs (array(),array())
        occurrence_df = occurrence_df.drop("label", axis=1)
        self.ocsvm.fit(occurrence_df)
        self.columns = occurrence_df.columns

    def __get_decision_points(self):
        Z = np.ones(
            (self.stacked_raster_coverages[1], self.stacked_raster_coverages[2]),
            dtype=np.float32,
        )
        Z *= self.configs["mask"][
            "no_data_val"
        ]  # This will be necessary to set points outside map to the minimum
        Z[self.inside_mask_idx[0], self.inside_mask_idx[1]] = self.ocsvm.predict(self.stacked_raster_coverages)
        return Z

    def generate(self, number_pseudo_absenses: int):
        pseudo_absenses_df = pd.DataFrame(self.columns)
        Z = self.__get_decision_points()
        x, y = np.where(Z < 0.5)
        x_y_chosed = []
        for _ in range(number_pseudo_absenses):
            while True:
                random_val = np.random.randint(len(x))
                matrix_position = (x[random_val], y[random_val])
                if matrix_position in x_y_chosed:
                    x_y_chosed.append(matrix_position)
                    break
            paseudo_absense_row = self.stacked_raster_coverages[
                :, x[random_val], y[random_val]
            ]
            pseudo_absenses_df.append(paseudo_absense_row)

        pseudo_absenses_df["label"] = 0
        return pseudo_absenses_df
