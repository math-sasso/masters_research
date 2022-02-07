from pathlib import Path
from .sources.soilgrids_downloader import SoilgridsDownloader
from easy_sdm.utils import PathUtils


class DownloadJob:
    def __init__(self) -> None:
        self.__build_empty_folders()

    def __build_empty_folders(self):
        raw_rasters_dir = Path.cwd() / "data/downloads/raw_rasters"
        folder_list = ["Bioclim", "Envirem", "Soilgrids"]
        for folder in folder_list:
            PathUtils.create_folder(raw_rasters_dir / folder)

    def download_bioclim_rasters(self):
        # TODO
        raise NotImplementedError()

    def download_envirem_rasters(self):
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
            / "data_processing"
            / "rasters"
            / "Bioclim_Rasters"
            / "bio1_annual_mean_temperature.tif"
        )
        root_dir = Path.cwd() / "data" / "raw_srasters" / "Soilgrids_Rasters"
        soilgrids_downloader = SoilgridsDownloader(
            reference_raster_path=reference_raster_path, root_dir=root_dir
        )
        soilgrids_downloader.set_soilgrids_requester(variable=variable)
        coverages = soilgrids_downloader.get_coverage_list()
        for cov in coverages:
            if coverage_filter in cov:
                soilgrids_downloader.download(coverage_type=cov)
