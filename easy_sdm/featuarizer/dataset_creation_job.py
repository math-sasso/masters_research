from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd
from easy_sdm.enums import ModellingType, PseudoSpeciesGeneratorType
from easy_sdm.enums.modelling_type import ModellingType
from easy_sdm.featuarizer.dataset_builder.VIF_calculator import VIFCalculator
from easy_sdm.typos import Species
from easy_sdm.utils import PathUtils
from sklearn.model_selection import train_test_split

from .dataset_builder.occurrence_dataset_builder import \
    OccurrancesDatasetBuilder
from .dataset_builder.pseudo_absense_dataset_builder import \
    PseudoAbsensesDatasetBuilder
from .dataset_builder.scaler import MinMaxScalerWrapper
from .dataset_builder.statistics_calculator import RasterStatisticsCalculator


class DatasetCreationJob:
    """
    [Create a dataset with species and pseudo spescies for SDM Machine Learning]
    """

    def __init__(
        self,
        raster_path_list: List[Path],
        ps_generator_type: PseudoSpeciesGeneratorType,
        ps_proportion: float,
        featuarizer_dirpath: Path,
        stacked_raster_coverages_path: Path,
        region_mask_raster_path: Path,
    ) -> None:

        self.raster_path_list = raster_path_list
        self.ps_generator_type = ps_generator_type
        self.ps_proportion = ps_proportion
        self.featuarizer_dirpath = featuarizer_dirpath
        self.stacked_raster_coverages_path = stacked_raster_coverages_path
        self.region_mask_raster_path = region_mask_raster_path
        self.raster_statistics_path = self.featuarizer_dirpath / "raster_statistics.csv"
        self.__build_empty_folders()
        self.__setup()

    def __build_empty_folders(self):
        PathUtils.create_folder(self.featuarizer_dirpath)

    def __create_statistics_dataset(self):
        raster_statistics_calculator = RasterStatisticsCalculator(
            raster_path_list=self.raster_path_list,
            mask_raster_path=self.region_mask_raster_path,
        )
        raster_statistics_calculator.build_table(
            output_path=self.raster_statistics_path
        )

        statistics_dataset = pd.read_csv(self.raster_statistics_path)

        return statistics_dataset

    def __setup(self):

        self.statistics_dataset = self.__create_statistics_dataset()

        self.occ_dataset_builder = OccurrancesDatasetBuilder(
            raster_path_list=self.raster_path_list,
        )
        self.min_max_scaler = MinMaxScalerWrapper(
            raster_path_list=self.raster_path_list,
            statistics_dataset=self.statistics_dataset,
        )

        self.raster_statistics_path = self.featuarizer_dirpath / "raster_statistics.csv"

        self.psa_dataset_builder = PseudoAbsensesDatasetBuilder(
            ps_generator_type=self.ps_generator_type,
            region_mask_raster_path=self.region_mask_raster_path,
            stacked_raster_coverages_path=self.stacked_raster_coverages_path,
        )

    def create_binary_classification_dataset(self, species_gdf: gpd.GeoDataFrame):

        occ_df = self.occ_dataset_builder.build(species_gdf)
        scaled_occ_df = self.min_max_scaler.scale_df(occ_df)

        number_pseudo_absenses = int(len(occ_df) * self.ps_proportion)
        psa_df = self.psa_dataset_builder.build(
            occurrence_df= scaled_occ_df, number_pseudo_absenses=number_pseudo_absenses
        )
        scaled_psa_df = self.min_max_scaler.scale_df(psa_df)
        scaled_df = pd.concat([scaled_occ_df, scaled_psa_df])

        return scaled_df


    def create_anomaly_detection_dataset(species_gdf: gpd.GeoDataFrame):
        # Avaliar se faz sentido criar subclasses
        raise NotImplementedError()

    def __split_dataset(self, df: pd.DataFrame,random_state:int):

        df_train,df_ = train_test_split(
            df, test_size=0.2, random_state=random_state
        )
        df_valid,df_test = train_test_split(
            df_, test_size=0.5, random_state=random_state
        )

        return df_train, df_valid, df_test

    def save_dataset(self,species:Species, df:pd.DataFrame ,modellting_type:ModellingType, random_state:int = 42):

        df_train,df_valid,df_test = self.__split_dataset(df,random_state)
        species_dataset_path = Path.cwd() / f"data/featuarizer/datasts/{species.get_name_for_paths()}/{modellting_type.value}"
        species_dataset_path.mkdir(parents=True, exist_ok=True)

        df_train.to_csv(species_dataset_path / "train.csv", index=False)
        df_valid.to_csv(species_dataset_path / "valid.csv", index=False)
        df_test.to_csv(species_dataset_path / "test.csv", index=False)

        vif_calculator = VIFCalculator(dataset_path=species_dataset_path / "train.csv", output_column='label')
        vif_calculator.calculate_vif()

        df_train_vif = df_train[vif_calculator.get_optimous_columns_with_label()]
        df_valid_vif = df_valid[vif_calculator.get_optimous_columns_with_label()]
        df_test_vif = df_test[vif_calculator.get_optimous_columns_with_label()]

        df_train_vif.to_csv(species_dataset_path / "vif_train.csv", index=False)
        df_valid_vif.to_csv(species_dataset_path / "vif_valid.csv", index=False)
        df_test_vif.to_csv(species_dataset_path / "vif_test.csv", index=False)

        vif_decision_df = vif_calculator.get_vif_df()
        vif_decision_df.to_csv(f"data/output/vif_analysis/vif_{species.get_name_for_paths()}", index=False)
