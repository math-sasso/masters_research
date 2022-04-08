from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import rasterio
from easy_sdm.configs import configs
from easy_sdm.ml import OCSVM


class BasePseudoSpeciesGenerator(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def fit(self, occurrence_data: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def generate(self, number_pseudo_absenses: int):
        raise NotImplementedError()


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

        self.hyperparameters = kwargs.get("hyperparameters", None)
        self.region_mask_raster = kwargs.get("region_mask_raster", None)
        self.stacked_raster_coverages = kwargs.get("stacked_raster_coverages", None)
        self.__check_arguments()
        self.configs = configs
        self.inside_mask_idx = np.where(
            self.region_mask_raster.read(1) != configs["mask"]["negative_mask_val"]
        )
        self.ocsvm = OCSVM(**self.hyperparameters)

    def __check_arguments(self):
        assert type(self.hyperparameters) is dict
        assert type(self.region_mask_raster) is rasterio.io.DatasetReader
        assert type(self.stacked_raster_coverages) is np.ndarray

    def fit(self, occurrence_df: pd.DataFrame):
        # Coords X and Y in two tuples where condition matchs (array(),array())

        occurrence_df = occurrence_df.drop("label", axis=1)
        self.ocsvm.fit(X_train=occurrence_df.values)
        self.columns = occurrence_df.columns

    def __get_decision_points(self):
        """[

            Z will be a 2D array with 3 possible values:
                # Outside Brazilian mask: -9999
                # Not valid predictions: -1 (Useful ones)
                # Valid predictions: 1
        ]

        Returns:
            [type]: [description]
        """
        Z = np.ones(
            (
                self.stacked_raster_coverages.shape[1],
                self.stacked_raster_coverages.shape[2],
            ),
            dtype=np.float32,
        )
        Z *= self.configs["mask"][
            "negative_mask_val"
        ]  # This will be necessary to set points outside map to the minimum

        inside_country_values = self.stacked_raster_coverages[
            :, self.inside_mask_idx[0], self.inside_mask_idx[1]
        ].T
        Z[self.inside_mask_idx[0], self.inside_mask_idx[1]] = self.ocsvm.predict(
            inside_country_values
        )

        return Z

    def generate(self, number_pseudo_absenses: int):
        pseudo_absenses_df = pd.DataFrame(columns=self.columns)
        Z = self.__get_decision_points()
        # Save Z because takes too long to run
        x, y = np.where(Z == -1)
        x_y_chosed = []
        for _ in range(number_pseudo_absenses):
            while True:
                random_val = np.random.randint(len(x))
                matrix_position = (x[random_val], y[random_val])
                if matrix_position not in x_y_chosed:
                    x_y_chosed.append(matrix_position)
                    break
            row_values = self.stacked_raster_coverages[:, x[random_val], y[random_val]]
            pseudo_absense_row = dict(zip(self.columns, row_values))
            pseudo_absenses_df = pseudo_absenses_df.append(
                pseudo_absense_row, ignore_index=True
            )
        return pseudo_absenses_df
