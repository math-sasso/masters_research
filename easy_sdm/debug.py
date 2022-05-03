from easy_sdm.utils import RasterLoader
from easy_sdm.visualization.debug_plots import RasterPlotter
from pathlib import Path


raw_raster = (
    RasterLoader("data/download/raw_rasters/Soilgrids/bdod/bdod_5-15cm_mean.tif")
    .load_dataset()
    .read(1)
)
processed_raster = (
    RasterLoader(
        "data/raster_processing/environment_variables_rasters/Soilgrids/bdod_0-5cm_mean.tif"
    )
    .load_dataset()
    .read(1)
)

import pdb

pdb.set_trace()
RasterPlotter.plot(processed_raster)
