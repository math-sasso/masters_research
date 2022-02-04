from typing import List
from easy_sdm.featuarizer import SDMDatasetCreator
from easy_sdm.enums import PseudoSpeciesGeneratorType
import pandas as pd
from pathlib import Path
from utils import PathUtils

# def save_raster_pat(tmp_path, mock_mask_raster_path,processed_raster_paths_list):

#     from easy_sdm.featuarizer import RasterStatisticsCalculator

#     output_path = tmp_path / "rasters_statistics.csv"
#     RasterStatisticsCalculator(
#         raster_path_list=processed_raster_paths_list, mask_raster_path=mock_mask_raster_path
#     ).build_table(output_path)

#     df_stats = pd.read_csv(output_path)

#     return df_stats

def build_sdm_dataset(
    ps_generator_type: str, ps_proportion: float, raster_path_list: List[str]
):

    ps_generator_type = {
        "RSEP": PseudoSpeciesGeneratorType.RSEP,
        "Random": PseudoSpeciesGeneratorType.RSEP.Random,
    }.get(ps_generator_type, f"{ps_generator_type}' is not supported!")

    statistics_dataset = pd.read_csv(
        Path.cwd() / "data/datasets/statistics_dataset.csv"
    )

    sdm_dataset_creator = SDMDatasetCreator(
        raster_path_list=raster_path_list,
        statistics_dataset=statistics_dataset,
        ps_generator_type=ps_generator_type,
        ps_proportion=ps_proportion,
    )

    df = sdm_dataset_creator.create_dataset()
    return df


# Ver o diagrama, essa parte vai ter que passsar pelo seletor
raster_path_list = PathUtils.get_rasters_filepaths_in_dir(
    Path.cwd() / "data/processed_rasters/standarized_rasters"
)
build_sdm_dataset(ps_generator_type='RSEP',
                   ps_proportion = 1,
                   raster_path_list=raster_path_list )
