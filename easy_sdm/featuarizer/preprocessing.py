from lib2to3.pgen2.pgen import DFAState
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from easy_sdm.configs import configs

from typing import List

# TODO:  Test this class
class RasterStatisticsCalculator:
    """[A class to extract basic statistics from rasters considering only the masked territorry]

    Attention: There is a problem in worldclim variables. They are not beeing well filtered in the south
    """

    def __init__(self, raster_path_list, mask_raster_path) -> None:
        self.configs = configs
        self.raster_path_list = raster_path_list
        self.mask_raster = rasterio.open(mask_raster_path)
        self.inside_mask_idx = np.where(
            self.mask_raster.read(1) != configs["maps"]["no_data_val"]
        )

    def build_table(self, output_path: Path):
        df = pd.DataFrame(
            columns=["raster_name", "min", "max", "mean", "std", "median"]
        )
        for raster_path in self.raster_path_list:
            raster = rasterio.open(raster_path)
            raster_data = raster.read(1)
            inside_mask_vec = raster_data[
                self.inside_mask_idx[1], self.inside_mask_idx[0]
            ]

            filtered_vec = inside_mask_vec[
                inside_mask_vec != configs["maps"]["no_data_val"]
            ]

            df = df.append(
                {
                    "raster_name": Path(raster_path).name,
                    "min": np.min(filtered_vec),
                    "max": np.max(filtered_vec),
                    "mean": np.mean(filtered_vec),
                    "std": np.std(filtered_vec),
                    "median": np.median(filtered_vec),
                },
                ignore_index=True,
            )

        df.to_csv(output_path, index=False)


# 1 - filtrar o brasil para pegar os minimos e maximos dentro do brasil apenas
# 2 - Com o raster em cima do brasil tirar estatisticas sem considerar no_data_val


# Tenho que pegar o array do raster de referencia
# idx = np.where(land_reference != no_data_val)
# raster_coverages_land = stacked_raster_coverages[:, idx[0], idx[1]].T
