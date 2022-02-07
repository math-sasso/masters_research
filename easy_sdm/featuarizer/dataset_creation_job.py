from typing import List
from easy_sdm.enums import PseudoSpeciesGeneratorType
import pandas as pd
from pathlib import Path
from utils import PathUtils
import geopandas as gpd

from .dataset_builder.occurrence_dataset_builder import OccurrancesDatasetBuilder
from .dataset_builder.pseudo_absense_dataset_builder  import PseudoAbsensesDatasetBuilder
from .dataset_builder.scaler import MinMaxScalerWrapper

class DatasetCreationJob:
    """[Create a dataset with species and pseudo spescies for SDM Machine Learning]"""

    def __init__(
        self,
        raster_path_list: List[Path],
        statistics_dataset: pd.DataFrame,
        ps_generator_type: PseudoSpeciesGeneratorType,
        ps_proportion: float,
    ) -> None:

        self.statistics_dataset = statistics_dataset
        self.raster_path_list = raster_path_list
        self.ps_generator_type = ps_generator_type
        self.ps_proportion = ps_proportion
        self.__setup()

    def __setup(self):
        self.occ_dataset_builder = OccurrancesDatasetBuilder(
            raster_path_list=self.raster_path_list
        )
        self.min_max_scaler = MinMaxScalerWrapper(
            raster_path_list=self.raster_path_list,
            statistics_dataset=self.statistics_dataset,
        )
        self.psa_dataset_builder = PseudoAbsensesDatasetBuilder(
            ps_generator_type=self.ps_generator_type
        )

    def create_dataset(self, species_gdf: gpd.GeoDataFrame):
        occ_df = self.occ_dataset_builder.build(species_gdf)
        number_pseudo_absenses = int(len(occ_df) * self.ps_proportion)

        scaled_occ_df = self.min_max_scaler.scale_df(occ_df)
        psa_df = self.psa_dataset_builder.build(
            occurrence_df=scaled_occ_df, number_pseudo_absenses=number_pseudo_absenses
        )
        scaled_psa_df = self.min_max_scaler.scale_df(psa_df)
        scaled_df = scaled_occ_df + scaled_psa_df
        import pdb; pdb.set_trace()
        return scaled_df

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
        Path.cwd() / "data/datasets/rasters_statistics.csv"
    )

    sdm_dataset_creator = SDMDatasetCreator(
        raster_path_list=raster_path_list,
        statistics_dataset=statistics_dataset,
        ps_generator_type=ps_generator_type,
        ps_proportion=ps_proportion,
    )

    species_gdf = gpd.read_file(Path.cwd() / "data/species_data/occurances/Canajus_cajan/Canajus_cajan.shp")

    df = sdm_dataset_creator.create_dataset(species_gdf=species_gdf)
    return df


# Ver o diagrama, essa parte vai ter que passsar pelo seletor
raster_path_list = PathUtils.get_rasters_filepaths_in_dir(
    Path.cwd() / "data/processed_rasters/standarized_rasters"
)
build_sdm_dataset(
    ps_generator_type="RSEP", ps_proportion=1, raster_path_list=raster_path_list
)
