import pickle
from pathlib import Path
from typing import List
from .environment_builder.relevant_rasters_selector import RelevantRastersSelector
from .environment_builder.environment_layer_stacker import EnverionmentLayersStacker
from easy_sdm.configs import configs
from easy_sdm.utils import PathUtils


class EnvironmentCreationJob:
    def __init__(self, output_dirpath: Path, all_rasters_path_list: List[Path]) -> None:
        self.output_dirpath = output_dirpath
        self.all_rasters_path_list = all_rasters_path_list
        self.relevant_rasters_selector = RelevantRastersSelector()
        self.env_layer_stacker = EnverionmentLayersStacker()
        self.__build_empty_folders()

    def __build_empty_folders(self):
        PathUtils.create_folder(self.output_dirpath)

    def build_environment(self):

        relevant_raster_path_list = self.relevant_rasters_selector.get_relevant_raster_path_list(
            raster_path_list=self.all_rasters_path_list,
        )

        self.relevant_rasters_selector.save_raster_list(
            raster_path_list=relevant_raster_path_list,
            output_path=self.output_dirpath / "relevant_raster_list",
        )

        self.env_layer_stacker.stack_and_save(
            raster_path_list=relevant_raster_path_list,
            output_path=self.output_dirpath / "environment_stack.npy",
        )
