import zipfile
from pathlib import Path

import rasterio
import requests
from easy_sdm.utils import PathUtils
from enums import RasterSource

from .initial_preprocessing.raster_clipper import RasterCliper
from .sources.soilgrids_downloader import SoilgridsDownloader


class Downloader:

    def __init__(self) -> None:
        pass

    def __request_multiple_tries(url:str, n_tries:int):
        i = 0
        while i<n_tries:
            response = requests.get(url)
            if response.status_code == 200:
                break
        return response


    def download(self, url:str, output_dirpath:str):
        response  = self.__request_multiple_tries(url,5)
        open(output_dirpath, "wb").write(response.content)


    def download_and_unzip(self, url:str, output_dirpath:str):
        response  = self.__request_multiple_tries(url,5)
        temp_dirpath = PathUtils.get_temp_dir()
        temp_zip = temp_dirpath / "temp.zip"
        open(temp_zip,"wb").write(response.content)
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dirpath)

class DownloadJob:
    def __init__(self, data_dirpath:Path) -> None:
        self.download_dirpath = data_dirpath / "download"
        self.__build_empty_folders()
        self.downloader = Downloader()
        self.raster_clipper = RasterCliper()

    def __build_empty_folders(self):

        folder_list = [e.name for e in RasterSource] + ["region_shapefile"]
        for folder in folder_list:
            PathUtils.create_folder(self.download_dirpath / folder)

    def download_bioclim_rasters(self):
        # TODO
        url = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_bio.zip"
        self.downloader.download_and_unzip(url,self.download_dirpath/RasterSource.Bioclim.name)

    def download_envirem_rasters(self):
        # File description format
        # [region] _ [time period] _ [circulation model] _ [resolution] _ [file format].zip
        # source downloads page
        # https://deepblue.lib.umich.edu/data/concern/data_sets/gt54kn05f
        #URLs
        temp_envirem_dirpath = PathUtils.get_temp_dir()
        url_part1 = "https://deepblue.lib.umich.edu/data/downloads/fq977t92p"
        url_part2 = "https://deepblue.lib.umich.edu/data/downloads/j67313902"
        url_part3 = "https://deepblue.lib.umich.edu/data/downloads/zc77sq25k"
        url_part4 = "https://deepblue.lib.umich.edu/data/downloads/w9505064f"
        self.downloader.download_and_unzip(url_part1,temp_envirem_dirpath)
        self.downloader.download_and_unzip(url_part2,temp_envirem_dirpath)
        self.downloader.download_and_unzip(url_part3,temp_envirem_dirpath)
        self.downloader.download_and_unzip(url_part4,temp_envirem_dirpath)

        for input_raster_path in PathUtils.get_rasters_filepaths_in_dir(temp_envirem_dirpath):
            output_path = self.download_dirpath/RasterSource.Envirem.name/input_raster_path.name
            self.raster_clipper.clip_and_save(source_raster=rasterio.open(input_raster_path),output_path=output_path)

    def download_shapefile_region(self):
        # TODO
        raise NotImplementedError()

    def download_soigrids_rasters(self, coverage_filter: str):
        variables = [
            "wrb",
            "cec",
            "clay",
            "phh2o",
            "silt",
            "ocs",
            "bdod",
            "cfvo",
            "nitrogen",
            "sand",
            "soc",
            "ocd",
        ]

        for variable in variables:
            self.__download_soilgrods_one_raster(variable, coverage_filter)

    def __download_soilgrods_one_raster(self, variable: str, coverage_filter: str):
        reference_raster_path = (
            Path.cwd()
            / "data"
            / "raster_processing"
            / "environment_variables_rasters"
            / RasterSource.Bioclim.name
            / "bio1_annual_mean_temperature.tif"
        )
        root_dir = self.raw_rasters_dirpath / RasterSource.Soilgrids.name
        soilgrids_downloader = SoilgridsDownloader(
            reference_raster_path=reference_raster_path, root_dir=root_dir
        )
        soilgrids_downloader.set_soilgrids_requester(variable=variable)
        coverages = soilgrids_downloader.get_coverage_list()
        for cov in coverages:
            if coverage_filter in cov:
                soilgrids_downloader.download(coverage_type=cov)
