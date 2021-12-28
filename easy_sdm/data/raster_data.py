from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from easy_sdm.configs import configs
from rasterio import features
from rasterio.warp import Resampling, calculate_default_transform, reproject


class RasterInfoExtractor(object):
    def __init__(self, raster: rasterio.io.DatasetReader):
        self.configs = configs
        self.__raster_array = self.__extrac_array(raster)
        self.__meta = self.__extract_meta(raster)

    def __extrac_array(self, raster: rasterio.io.DatasetReader):
        self.__one_layer_verifier(raster)
        raster_array = raster.read(1)
        return raster_array

    def __one_layer_verifier(self, raster: rasterio.io.DatasetReader):
        if raster.meta["count"] > 1:
            raise Exception("Raster images are suppose to have only one layer")
        elif raster.meta["count"] == 0:
            raise Exception("For some reason this raster is empty")

    def __extract_meta(self, raster: rasterio.io.DatasetReader):
        return raster.meta

    def get_array(self):
        return self.__raster_array

    def get_driver(self):
        return self.__meta["GTiff"]

    def get_data_type(self):
        return self.__meta["dtype"]

    def get_nodata_val(self):
        return self.__meta["nodata"]

    def get_width(self):
        return self.__meta["width"]

    def get_heigh(self):
        return self.__meta["heigh"]

    def get_crs(self):
        return self.__meta["crs"]

    def get_affine(self):
        return self.__meta["transform"]

    def get_resolution(self):
        return abs(self.__meta["transform"][0])

    def get_xgrid(self):
        xgrid = np.arange(
            self.configs["maps"]["brazil_limits_with_security"]["west"],
            self.configs["maps"]["brazil_limits_with_security"]["west"]
            + self.__raster_array.shape[1] * self.get_resolution(),
            self.get_resolution(),
        )
        return xgrid

    def get_ygrid(self):
        ygrid = np.arange(
            configs["maps"]["brazil_limits_with_security"]["south"],
            configs["maps"]["brazil_limits_with_security"]["south"]
            + self.__raster_array.shape[0] * self.get_resolution(),
            self.get_resolution(),
        )
        return ygrid

    def get_xcenter(self):
        xgrid = self.get_xgrid()
        return np.mean(xgrid)

    def get_ycenter(self):
        ygrid = self.get_ygrid()
        return np.mean(ygrid)


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

    def clip_and_save(
        self, source_raster: rasterio.io.DatasetReader, output_path: Path
    ):
        result_profile, cropped_raster_matrix = self.clip(source_raster)
        with rasterio.open(output_path, "w", **result_profile) as dst:
            dst.write(cropped_raster_matrix, 1)


class RasterShapefileBurner:
    def __init__(self, reference_raster: rasterio.io.DatasetReader):
        meta = reference_raster.meta.copy()
        meta.update(compress="lzw")
        self.meta = meta

    def burn_and_save(self, shapfile: gpd.GeoDataFrame, output_path: Path):
        """Inspired for #https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python

        Args:
            shapfile (gpd.GeoDataFrame): [description]
            output_path (str): [description]
        """

        with rasterio.open(output_path, "w+", **self.meta) as out:
            out_arr = out.read(1)

            # this is where we create a generator of geom, value pairs to use in rasterizing
            shapes = ((geom, 0) for geom in shapfile.geometry)

            burned = rasterio.features.rasterize(
                shapes=shapes, fill=0, out=out_arr,all_touched=True, transform=out.transform
            )

            out.write_band(1, burned)


class RasterStandarizer:
    def __init__(self) -> None:
        raise NotImplementedError("Not implemented yet")

    def standarize(raw_raster):
        import os.path
        import time

        while not os.path.exists(file_path):
            time.sleep(1)

        if os.path.isfile(file_path):
            pass
        else:
            raise ValueError("%s isn't a file!" % file_path)


class RasterValuesStandarizer:
    def __init__(self) -> None:
        raise NotImplementedError("Not implemented yet")

    def __set_to_float32(self, profile, cropped_data):
        if profile["dtype"] != np.float32:
            # data = data.astype(rasterio.float32)
            cropped_data = np.float32(cropped_data)
            profile["dtype"] = np.float32

        return profile, cropped_data

    def __set_no_data_val(self, profile, cropped_data):
        if profile["nodata"] != configs["maps"]["no_data_val"]:
            cropped_data = np.where(
                cropped_data < configs["maps"]["no_data_val"],
                configs["maps"]["no_data_val"],
                cropped_data,
            )
            profile["nodata"] = configs["maps"]["no_data_val"]

        return profile, cropped_data


class RasterCRSStandarizer:
    def __init__(self) -> None:
        self.configs = configs
        raise NotImplementedError("Not implemented yet")

    def reproject_crs(self, input_raster_path: Path, output_raster_path: Path):
        dst_crs = str(configs["maps"]["default_epsg"])
        with rasterio.open(input_raster_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": dst_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )

            with rasterio.open(output_raster_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )
