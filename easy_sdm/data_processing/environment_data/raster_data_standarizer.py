from pathlib import Path

import numpy as np
from easy_sdm.configs import configs
from utils import RasterUtils


class RasterDataStandarizer:
    """[A class to perform expected standarizations]"""

    def __init__(self,) -> None:
        self.configs = configs

    def standarize(self, raster, output_path: Path):
        profile = raster.profile.copy()
        data = raster.read(1)
        width, height = data.shape
        data = np.float32(data)
        data = np.where(
            data == profile["nodata"], configs["maps"]["no_data_val"], data,
        )

        profile.update(
            {
                "driver": "GTiff",
                "count": 1,
                "crs": {"init": f"EPSG:{configs['maps']['default_epsg']}"},
                "width": width,
                "height": height,
                "nodata": configs["maps"]["no_data_val"],
                "dtype": np.float32,
            }
        )
        RasterUtils.save_raster(data=data, profile=profile, output_path=output_path)
