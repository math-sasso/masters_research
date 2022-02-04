import tempfile
from pathlib import Path
from urllib import response

import geopandas as gpd
import numpy as np
import rasterio
from easy_sdm.configs import configs
from easy_sdm.utils import PathUtils, RasterUtils, TemporaryDirectory
from easy_sdm.data import RasterLoader, ShapefileLoader
from rasterio import features
from rasterio.warp import Resampling, calculate_default_transform, reproject
from owslib.wcs import WebCoverageService


class SoilgridsDownloader:
    """[summary]

    References:
    - The soilgrids lib:
    - The possible soilgrids variables: https://maps.isric.org/
    """

    def __init__(self, reference_raster_path: Path, root_dir: Path) -> None:

        self.reference_raster = RasterLoader(
            raster_path=reference_raster_path
        ).load_dataset()
        self.root_dir = root_dir
        self.soilgrids_requester = None
        self.variable = None
        self.configs = configs
        self.width = self.reference_raster.width
        self.height = self.reference_raster.height

    def set_soilgrids_requester(self, variable: str):
        url = f"http://maps.isric.org/mapserv?map=/map/{variable}.map"
        self.soilgrids_requester = WebCoverageService(url, version="1.0.0")
        self.variable = variable

    def __check_if_soilgrids_requester(self):
        if self.soilgrids_requester is None:
            raise ValueError("Please set soilgrids_requester_before")

    def __build_bbox(self):
        # check bounding box
        limits = self.configs["maps"]["brazil_limits_with_security"]
        if limits["west"] > limits["east"] or limits["south"] > limits["north"]:
            raise ValueError(
                "Please provide valid bounding box values for west, east, south and north."
            )
        else:
            bbox = (limits["west"], limits["south"], limits["east"], limits["north"])
        return bbox

    def download(
        self, coverage_type: str,
    ):
        self.__check_if_soilgrids_requester()
        output_dir = self.root_dir / self.variable
        PathUtils.create_folder(output_dir)
        coverage_type = coverage_type.replace(".", "_")
        output_path = output_dir / f"{coverage_type}.tif"
        if not Path(output_path).is_file():
            crs = "urn:ogc:def:crs:EPSG::4326"
            response = None
            i = 0
            while response is None and i < 5:
                try:
                    response = self.soilgrids_requester.getCoverage(
                        identifier=coverage_type,
                        crs=crs,
                        bbox=self.__build_bbox(),
                        resx=None,
                        resy=None,
                        width=self.width,
                        height=self.height,
                        response_crs=crs,
                        format="GEOTIFF_INT16",
                    )  # bytes
                    RasterUtils.save_binary_raster(response, output_path)
                except Exception as e:
                    print(type(e))

            i += 1

    def get_coverage_list(self):
        self.__check_if_soilgrids_requester()
        return list(self.soilgrids_requester.contents)


class RasterInfoExtractor:
    """[A Wrapper to extract relevant information from raster objects]"""

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
            raise ValueError("Raster images are suppose to have only one layer")
        elif raster.meta["count"] == 0:
            raise ValueError("For some reason this raster is empty")

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


class RasterShapefileBurner:
    """[Burn a shapefile troought a reference raster. The reference is used only to get the Affine properties]
    Inside Shapefile: 0
    Outside Shapefile: -9999. (no_data_val)
    """

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
                shapes=shapes,
                fill=0,
                out=out_arr,
                all_touched=True,
                transform=out.transform,
            )

            out.write_band(1, burned)


class RasterCliper:
    """
    [Clip a raster trought a configured square]

    """

    def __init__(self):
        pass

    def __get_window_from_extent(self, aff):
        """Get a portion form a raster array based on the country limits"""
        map_limits = configs["maps"]["brazil_limits_with_security"]
        col_start, row_start = ~aff * (map_limits["west"], map_limits["north"],)
        col_stop, row_stop = ~aff * (map_limits["east"], map_limits["south"],)
        return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))

    def __create_profile(
        self,
        source_raster: rasterio.io.DatasetReader,
        cropped_raster_matrix: np.ndarray,
    ):
        """[Create profile with required changes to clip rasters]
        Affine logic: (resolution_x,rot1,extreme_west_point,rot2,resolution_y(negative),extreme_north_point)
        Example: Affine(0.008333333333333333, 0.0, -180.0, 0.0, -0.008333333333333333, 90.0)
        """
        map_limits = configs["maps"]["brazil_limits_with_security"]
        result_profile = source_raster.profile.copy()
        cropped_data = cropped_raster_matrix.copy()
        src_transfrom = result_profile["transform"]
        result_transfrom = rasterio.Affine(
            src_transfrom[0],
            src_transfrom[1],
            map_limits["west"],
            src_transfrom[3],
            src_transfrom[4],
            map_limits["north"],
        )

        result_profile.update(
            {
                "width": cropped_data.shape[1],
                "height": cropped_data.shape[0],
                "transform": result_transfrom,
            }
        )

        return result_profile

    def clip(self, source_raster: rasterio.io.DatasetReader):
        window_region = self.__get_window_from_extent(source_raster.meta["transform"])
        cropped_raster_matrix = source_raster.read(1, window=window_region)
        profile = self.__create_profile(source_raster, cropped_raster_matrix)
        return profile, cropped_raster_matrix

    def clip_and_save(
        self, source_raster: rasterio.io.DatasetReader, output_path: Path
    ):
        result_profile, cropped_raster_matrix = self.clip(source_raster)
        RasterUtils.save_raster(
            data=cropped_raster_matrix, profile=result_profile, output_path=output_path
        )


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


class RasterStandarizer:
    """[A class that centralizes all RasterStandarization applications and take control over it]"""

    def __init__(self) -> None:
        self.configs = configs

    def standarize_soilgrids(self, input_path: Path, output_path: Path):
        PathUtils.create_folder(output_path.parents[0])
        RasterDataStandarizer().standarize(
            raster=rasterio.open(input_path), output_path=output_path
        )

    def standarize_bioclim_envirem(self, input_path: Path, output_path: Path):
        tempdir = TemporaryDirectory()
        raster_cliped_path = Path(tempdir.name) / "raster_cliped.tif"
        RasterCliper().clip_and_save(
            source_raster=rasterio.open(input_path), output_path=raster_cliped_path
        )
        PathUtils.create_folder(output_path.parents[0])
        RasterDataStandarizer().standarize(
            raster=rasterio.open(raster_cliped_path), output_path=output_path
        )


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
