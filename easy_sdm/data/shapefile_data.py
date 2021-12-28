from pathlib import Path

import geopandas as gpd
from easy_sdm.data import ShapefileLoader


class ShapefileRegion:
    def __init__(self, shapefile_path: Path):
        self.gdf_region = ShapefileLoader(shapefile_path).load_dataset()

    def get_points_inside(self, gdf: gpd.GeoDataFrame):
        """get if points are inside dataframe
        Inspired for :https://www.matecdev.com/posts/point-in-polygon.html

        Args:
            df ([pd.DataFrame]): dataframe with lat lon values
        """
        points_in_polygon = gdf.sjoin(self.gdf_region, predicate="within", how="inner")
        new_gdf = points_in_polygon[list(gdf.columns)]
        new_gdf = new_gdf.reset_index(drop=True)
        return new_gdf
