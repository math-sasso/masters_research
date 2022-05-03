from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd
from easy_sdm.enums import ModellingType, PseudoSpeciesGeneratorType
from easy_sdm.enums.modelling_type import ModellingType
from easy_sdm.featuarizer.dataset_builder.VIF_calculator import VIFCalculator
from easy_sdm.typos import Species
from easy_sdm.utils import PathUtils, PickleLoader
from sklearn.model_selection import train_test_split

from .dataset_builder.occurrence_dataset_builder import OccurrancesDatasetBuilder
from .dataset_builder.pseudo_absense_dataset_builder import PseudoAbsensesDatasetBuilder
from .dataset_builder.scaler import MinMaxScalerWrapper
from .dataset_builder.statistics_calculator import RasterStatisticsCalculator


class DatasetCreationJob:
    """
    [Create a dataset with species and pseudo spescies for SDM Machine Learning]
    """

    def __init__(
        self,
        root_data_dirpath: Path,
        ps_proportion: float,
        ps_generator_type: PseudoSpeciesGeneratorType,
    ) -> None:

        self.inference_proportion_from_all_data = 0.2
        self.test_proportion_from_inference_data = 0.5
        self.ps_proportion = ps_proportion
        self.ps_generator_type = ps_generator_type
        self.root_data_dirpath = root_data_dirpath
        self.__setup()
        self.__build_empty_folders()

    def __build_empty_folders(self):
        PathUtils.create_folder(self.featuarizer_dirpath)

    def __create_statistics_dataset(self):
        raster_statistics_calculator = RasterStatisticsCalculator(
            raster_path_list=self.raster_path_list,
            mask_raster_path=self.region_mask_raster_path,
        )
        raster_statistics_calculator.build_table(
            output_path=self.featuarizer_dirpath / "raster_statistics.csv"
        )

        statistics_dataset = pd.read_csv(
            self.featuarizer_dirpath / "raster_statistics.csv"
        )

        return statistics_dataset

    def __setup(self):

        self.raster_path_list_path = (
            self.root_data_dirpath / "environment/relevant_raster_list"
        )
        self.featuarizer_dirpath = self.root_data_dirpath / "featuarizer"
        self.stacked_raster_coverages_path = (
            self.root_data_dirpath / "environment/environment_stack.npy"
        )
        self.region_mask_raster_path = (
            self.root_data_dirpath / "raster_processing/region_mask.tif"
        )

        self.raster_path_list = PickleLoader(self.raster_path_list_path).load_dataset()
        self.statistics_dataset = self.__create_statistics_dataset()

        self.occ_dataset_builder = OccurrancesDatasetBuilder(
            raster_path_list=self.raster_path_list,
        )
        self.min_max_scaler = MinMaxScalerWrapper(
            raster_path_list=self.raster_path_list,
            statistics_dataset=self.statistics_dataset,
        )

    def create_general_dataset(
        self, species_gdf: gpd.GeoDataFrame,
    ):

        self.psa_dataset_builder = PseudoAbsensesDatasetBuilder(
            ps_generator_type=self.ps_generator_type,
            region_mask_raster_path=self.region_mask_raster_path,
            stacked_raster_coverages_path=self.stacked_raster_coverages_path,
        )

        self.occ_dataset_builder.build(species_gdf)
        occ_df = self.occ_dataset_builder.get_dataset()
        coords_occ_df = self.occ_dataset_builder.get_coordinates_df()
        scaled_occ_df = self.min_max_scaler.scale_df(occ_df)

        number_pseudo_absenses = int(len(occ_df) * self.ps_proportion)
        self.psa_dataset_builder.build(
            occurrence_df=scaled_occ_df, number_pseudo_absenses=number_pseudo_absenses
        )
        psa_df = self.psa_dataset_builder.get_dataset()
        coords_psa_df = self.psa_dataset_builder.get_coordinates_df()
        scaled_psa_df = self.min_max_scaler.scale_df(psa_df)
        scaled_df = pd.concat([scaled_occ_df, scaled_psa_df], ignore_index=True)
        coords_df = pd.concat([coords_occ_df, coords_psa_df], ignore_index=True)

        return scaled_df, coords_df

    def __split_dataset(
        self, df: pd.DataFrame, random_state: int, modellting_type: ModellingType
    ):

        # spliting between train and inference
        if modellting_type == ModellingType.BinaryClassification:
            df_train, df_ = train_test_split(
                df,
                test_size=self.inference_proportion_from_all_data,
                random_state=random_state,
            )

        elif modellting_type == ModellingType.AnomalyDetection:
            import pdb

            pdb.set_trace()
            df_occ = df[df["label"] == 1]
            df_psa = df[df["label"] == 0]
            num_inference_data = len(df_train) * self.inference_proportion_from_all_data
            df_train = df_occ[num_inference_data:].reset_index()
            df_inference_occ = df_occ[:num_inference_data]
            df_inference_psa = df_psa[: len(df_inference_occ)]
            df_ = pd.concat([df_inference_psa, df_inference_occ], ignore_index=True)
            df_.index = pd.RangeIndex(len(df_train), len(df_train) + len(df_))
        # spliting between validation and test
        df_valid, df_test = train_test_split(
            df_,
            test_size=self.test_proportion_from_inference_data,
            random_state=random_state,
        )

        return df_train, df_valid, df_test

    def _split_coords_df_dataset_as_sdm_dataset(
        self,
        coords_df: pd.DataFrame,
        sdm_df_train: pd.DataFrame,
        sdm_df_valid: pd.DataFrame,
        sdm_df_test: pd.DataFrame,
    ):
        coords_df_train = coords_df.iloc[list(sdm_df_train.index)]
        coords_df_valid = coords_df.iloc[list(sdm_df_valid.index)]
        coords_df_test = coords_df.iloc[list(sdm_df_test.index)]
        return coords_df_train, coords_df_valid, coords_df_test

    def save_dataset(
        self,
        species: Species,
        sdm_df: pd.DataFrame,
        coords_df: pd.DataFrame,
        modellting_type: ModellingType,
        random_state: int = 42,
    ):

        df_train, df_valid, df_test = self.__split_dataset(
            sdm_df, random_state, modellting_type
        )
        (
            coords_df_train,
            coords_df_valid,
            coords_df_test,
        ) = self._split_coords_df_dataset_as_sdm_dataset(
            coords_df, df_train, df_valid, df_test
        )

        species_dataset_path = (
            self.root_data_dirpath
            / f"featuarizer/datasets/{species.get_name_for_paths()}/{modellting_type.value}"
        )
        PathUtils.create_folder(species_dataset_path)

        species_dataset_path.mkdir(parents=True, exist_ok=True)

        coords_df_train.to_csv(species_dataset_path / "train_coords.csv", index=False)
        coords_df_valid.to_csv(species_dataset_path / "valid_coords.csv", index=False)
        coords_df_test.to_csv(species_dataset_path / "test_coords.csv", index=False)

        df_train.to_csv(species_dataset_path / "train.csv", index=False)
        df_valid.to_csv(species_dataset_path / "valid.csv", index=False)
        df_test.to_csv(species_dataset_path / "test.csv", index=False)

        vif_calculator = VIFCalculator(
            dataset_path=species_dataset_path / "train.csv", output_column="label"
        )
        vif_calculator.calculate_vif()

        df_train_vif = df_train[vif_calculator.get_optimous_columns_with_label()]
        df_valid_vif = df_valid[vif_calculator.get_optimous_columns_with_label()]
        df_test_vif = df_test[vif_calculator.get_optimous_columns_with_label()]

        df_train_vif.to_csv(species_dataset_path / "vif_train.csv", index=False)
        df_valid_vif.to_csv(species_dataset_path / "vif_valid.csv", index=False)
        df_test_vif.to_csv(species_dataset_path / "vif_test.csv", index=False)

        vif_decision_df = vif_calculator.get_vif_df()
        vif_decision_df.to_csv(
            species_dataset_path / f"vif_{species.get_name_for_paths()}.csv",
            index=False,
        )
