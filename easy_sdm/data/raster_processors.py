import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from configs import configs


class RasterCliper:
    def __init__(self):
        pass

    def __get_window_from_extent(self, aff):
        """ "Get a portion form a raster array based on the country limits"""
        map_limits = configs["maps"]["brazil_limits_with_security"]
        col_start, row_start = ~aff * (
            map_limits["west"],
            map_limits["north"],
        )
        col_stop, row_stop = ~aff * (
            map_limits["east"],
            map_limits["south"],
        )
        return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))

    def __data_conversions(
        self,
        source_raster: rasterio.io.DatasetReader,
        cropped_raster_matrix: np.ndarray,
    ):
        map_limits = configs["maps"]["brazil_limits_with_security"]
        result_profile = source_raster.profile.copy()
        cropped_data = cropped_raster_matrix.copy()

        # 0 - Source Profile
        resolution = result_profile["transform"][0]
        rot1 = result_profile["transform"][1]
        _ = result_profile["transform"][2]  # x_init_point
        rot2 = result_profile["transform"][3]
        n_resolution = result_profile["transform"][4]
        _ = result_profile["transform"][5]  # y_init_point

        # 1 - Converting data to np.float32
        if result_profile["dtype"] != np.float32:
            # data = data.astype(rasterio.float32)
            cropped_data = np.float32(cropped_data)
            result_profile["dtype"] = np.float32

        # 2 - Converting no data to -9999.0
        # Note that form brazilian mask. The background is 0. So it wont be affected
        # Here we have an attention point. This < -9999 could result in sensible defaults
        if result_profile["nodata"] != configs["maps"]["no_data_val"]:
            cropped_data = np.where(
                cropped_data < configs["maps"]["no_data_val"],
                configs["maps"]["no_data_val"],
                cropped_data,
            )
            result_profile["nodata"] = configs["maps"]["no_data_val"]

        # 3 - Changing width and height to the cropped region
        result_profile["width"] = cropped_data.shape[1]
        result_profile["height"] = cropped_data.shape[0]

        # 4 - setting CRS
        default_epsg = configs["maps"]["default_epsg"]
        result_profile["crs"] = {"init": f"EPSG:{default_epsg}"}

        # 5 - Changing Affine parameters
        # Example: Affine(0.008333333333333333, 0.0, -180.0, 0.0, -0.008333333333333333, 90.0)
        result_profile["transform"] = rasterio.Affine(
            resolution,
            rot1,
            map_limits["west"],
            rot2,
            n_resolution,
            map_limits["north"],
        )

        return result_profile

    def clip(self, source_raster: rasterio.io.DatasetReader):
        window_region = self.__get_window_from_extent(source_raster.meta["transform"])
        cropped_raster_matrix = source_raster.read(1, window=window_region)
        profile = self.__data_conversions(source_raster, cropped_raster_matrix)
        return profile, cropped_raster_matrix

    def clip_and_save(self, source_raster: rasterio.io.DatasetReader, output_path: str):
        result_profile, cropped_raster_matrix = self.clip(source_raster)
        with rasterio.open(output_path, "w", **result_profile) as dst:
            dst.write(cropped_raster_matrix, 1)


class RasterShapefileBurner:
    def __init__(self, reference_raster: rasterio.io.DatasetReader):
        meta = reference_raster.meta.copy()
        meta.update(compress="lzw")
        self.meta = meta

    
    def burn_and_save(self, shapfile: gpd.GeoDataFrame, output_path: str):
        """Inspired for #https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python

        Args:
            shapfile (gpd.GeoDataFrame): [description]
            output_path (str): [description]
        """

        with rasterio.open(output_path, "w+", **self.meta) as out:
            out_arr = out.read(1)
    
            # this is where we create a generator of geom, value pairs to use in rasterizing
            shapes = (
                (geom, 0)
                for geom in shapfile.geometry
            )
    
            burned = rasterio.features.rasterize(
                shapes=shapes, fill=0, out=out_arr, transform=out.transform
            )
      
            out.write_band(1, burned)