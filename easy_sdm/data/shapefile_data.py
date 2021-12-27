from pathlib import Path

import geopandas as gpd
from utils.utils import timeit

from data.data_loader import ShapefileLoader


class ShapefileRegion:
    def __init__(self, shapefile_path: Path):
        self.gdf_region = ShapefileLoader().read_geodataframe(shapefile_path)

    @timeit
    def get_points_inside(self, gdf: gpd.GeoDataFrame):
        """get if points are inside dataframe
        Inspired for :https://www.matecdev.com/posts/point-in-polygon.html

        Args:
            df ([pd.DataFrame]): dataframe with lat lon values
        """
        gdf["coords"] = gdf["geometry"]
        points_in_polygon = gpd.tools.sjoin(
            gdf, self.gdf_region, predicate="within", how="left"
        )
        new_gdf = points_in_polygon[list(gdf.columns)]
        new_gdf = new_gdf.drop(["coords"], axis=1)
        return new_gdf
