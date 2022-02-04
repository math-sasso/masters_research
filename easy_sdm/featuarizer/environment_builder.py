import gc
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from easy_sdm.configs import configs
from easy_sdm.data import RasterInfoExtractor, RasterLoader


class EnverionmentLayersStacker:
    """[Create a 3D array from 2D arrays stacked]
    """

    def __init__(self) -> None:
        pass

    def load(self, input_path: Path):
        return np.load(input_path)

    def stack_and_save(self, raster_path_list: List[Path], output_path: Path):
        if not str(output_path).endswith(".npy"):
            raise TypeError("output_path must ends with .npy")

        coverage = self.stack(raster_path_list)
        with open(output_path, "wb") as f:
            np.save(f, coverage)

        del coverage
        gc.collect()

    def stack(self, raster_path_list: List[Path]):

        all_env_values_list = []
        for path in raster_path_list:
            raster = RasterLoader(path).load_dataset()
            raster_array = RasterInfoExtractor(raster).get_array()
            all_env_values_list.append(raster_array)
            del raster
            del raster_array
            gc.collect()

        coverage = np.stack(all_env_values_list)

        return coverage
