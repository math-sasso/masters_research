from pathlib import Path
from typing import List

from sqlalchemy import true

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
from .dataset_builder.statistics_calculator import (
    RasterStatisticsCalculator,
    DataframeStatisticsCalculator,
)


class DatasetCreationJob:
    """
    [Create a dataset with species and pseudo spescies for SDM Machine Learning]
    """

    def __init__(
        self,
        root_data_dirpath: Path,
        species: Species,
        species_gdf: gpd.GeoDataFrame,
        ps_proportion: float,
        ps_generator_type: PseudoSpeciesGeneratorType,
        modelling_type: ModellingType,
    ) -> None:

        self.inference_proportion_from_all_data = 0.2
        self.test_proportion_from_inference_data = 0.5

        self.root_data_dirpath = root_data_dirpath
        self.species = species
        self.species_gdf = species_gdf
        self.ps_proportion = ps_proportion
        self.ps_generator_type = ps_generator_type
        self.modelling_type = modelling_type
        self.random_state = 42

        self.__setup()
        self.__build_empty_folders()

        self.train_occ = None
        self.val_occ = None
        self.test_occ = None

        self.train_psa = None
        self.val_psa = None
        self.test_psa = None

    def __build_empty_folders(self):
        PathUtils.create_folder(self.featuarizer_dirpath)

    def __setup(self):

        self.species_dataset_path = (
            self.root_data_dirpath
            / f"featuarizer/datasets/{self.species.get_name_for_paths()}/{self.modelling_type.value}"
        )
        self.species_dataset_path.mkdir(parents=True, exist_ok=True)

        self.raster_path_list_path = (
            self.root_data_dirpath / "environment/relevant_raster_list"
        )
        self.featuarizer_dirpath = self.root_data_dirpath / "featuarizer"

        self.region_mask_raster_path = (
            self.root_data_dirpath / "raster_processing/region_mask.tif"
        )

        self.raster_path_list = PickleLoader(self.raster_path_list_path).load_dataset()

        statistics_dataset = (
            self.__create_statistics_dataset()
        )  # apenas a patir do treino
        self.min_max_scaler = MinMaxScalerWrapper(
            statistics_dataset=statistics_dataset,
        )

        self.occ_dataset_builder = OccurrancesDatasetBuilder(
            raster_path_list=self.raster_path_list,
        )

    def __split_dataset(
        self, df: pd.DataFrame,
    ):

        df_train, df_ = train_test_split(
            df,
            test_size=self.inference_proportion_from_all_data,
            random_state=self.random_state,
        )
        df_valid, df_test = train_test_split(
            df_,
            test_size=self.test_proportion_from_inference_data,
            random_state=self.random_state,
        )

        return df_train, df_valid, df_test

    def __reflect_split_from_other_dataset(
        self,
        input_df: pd.DataFrame,
        sdm_df_train: pd.DataFrame,
        sdm_df_valid: pd.DataFrame,
        sdm_df_test: pd.DataFrame,
    ):
        input_df_train = input_df.iloc[list(sdm_df_train.index)]
        input_df_valid = input_df.iloc[list(sdm_df_valid.index)]
        input_df_test = input_df.iloc[list(sdm_df_test.index)]
        return input_df_train, input_df_valid, input_df_test

    def __create_statistics_dataset(self):

        # raster_statistics_calculator = DataframeStatisticsCalculator(
        #     df=df,
        # )
        # statistics_dataset_path = self.species_dataset_path / 'statistics.csv'

        statistics_dataset_path = (
            self.root_data_dirpath / f"featuarizer/raster_statistics.csv"
        )

        raster_statistics_calculator = RasterStatisticsCalculator(
            raster_path_list=self.raster_path_list,
            mask_raster_path=self.region_mask_raster_path,
        )
        raster_statistics_calculator.build_table(statistics_dataset_path)

        statistics_dataset = pd.read_csv(statistics_dataset_path)

        return statistics_dataset

    def __create_occ_df(self):

        self.occ_dataset_builder.build(self.species_gdf)
        occ_df = self.occ_dataset_builder.get_dataset()
        coords_occ_df = self.occ_dataset_builder.get_coordinates_df()

        scaled_occ_df = self.min_max_scaler.scale_df(occ_df)
        (
            scaled_train_occ_df,
            scaled_val_occ_df,
            scaled_test_occ_df,
        ) = self.__split_dataset(scaled_occ_df)

        (
            train_coords_occ_df,
            val_coords_occ_df,
            test_coords_occ_df,
        ) = self.__reflect_split_from_other_dataset(
            coords_occ_df, scaled_train_occ_df, scaled_val_occ_df, scaled_test_occ_df
        )

        self.scaled_train_occ_df = scaled_train_occ_df
        self.scaled_val_occ_df = scaled_val_occ_df
        self.scaled_test_occ_df = scaled_test_occ_df

        self.train_coords_occ_df = train_coords_occ_df
        self.val_coords_occ_df = val_coords_occ_df
        self.test_coords_occ_df = test_coords_occ_df

    def __create_psa_df(self,):
        self.psa_dataset_builder = PseudoAbsensesDatasetBuilder(
            root_data_dirpath=self.root_data_dirpath,
            ps_generator_type=self.ps_generator_type,
            scaled_occurrence_df=self.scaled_train_occ_df,
            min_max_scaler=self.min_max_scaler,
        )

        occ_df_size = (
            len(self.scaled_train_occ_df)
            + len(self.scaled_val_occ_df)
            + len(self.scaled_test_occ_df)
        )

        number_pseudo_absenses = int(occ_df_size * self.ps_proportion)
        self.psa_dataset_builder.build(number_pseudo_absenses=number_pseudo_absenses)

        psa_df = self.psa_dataset_builder.get_dataset()
        coords_psa_df = self.psa_dataset_builder.get_coordinates_df()

        scaled_psa_df = self.min_max_scaler.scale_df(psa_df)
        (
            scaled_train_psa_df,
            scaled_val_psa_df,
            scaled_test_psa_df,
        ) = self.__split_dataset(scaled_psa_df)

        (
            train_coords_psa_df,
            val_coords_psa_df,
            test_coords_psa_df,
        ) = self.__reflect_split_from_other_dataset(
            coords_psa_df, scaled_train_psa_df, scaled_val_psa_df, scaled_test_psa_df
        )

        # Estao vindo valores -9999.0
        self.scaled_train_psa_df = scaled_train_psa_df
        self.scaled_val_psa_df = scaled_val_psa_df
        self.scaled_test_psa_df = scaled_test_psa_df

        self.train_coords_psa_df = train_coords_psa_df
        self.val_coords_psa_df = val_coords_psa_df
        self.test_coords_psa_df = test_coords_psa_df

    def __create_binary_classification_dataset(self):
        self.scaled_train_df = pd.concat(
            [self.scaled_train_occ_df, self.scaled_train_psa_df], ignore_index=True
        )
        self.scaled_val_df = pd.concat(
            [self.scaled_val_occ_df, self.scaled_val_psa_df], ignore_index=True
        )
        self.scaled_test_df = pd.concat(
            [self.scaled_test_occ_df, self.scaled_test_psa_df], ignore_index=True
        )

        self.train_coords_df = pd.concat(
            [self.train_coords_occ_df, self.train_coords_psa_df], ignore_index=True
        )
        self.val_coords_df = pd.concat(
            [self.val_coords_occ_df, self.val_coords_psa_df], ignore_index=True
        )
        self.test_coords_df = pd.concat(
            [self.test_coords_occ_df, self.test_coords_psa_df], ignore_index=True
        )

    def __create_anomaly_detection_dataset(self):
        self.scaled_train_df = self.scaled_train_occ_df.reset_index(drop=True)
        self.scaled_val_df = pd.concat(
            [self.scaled_val_occ_df, self.scaled_val_psa_df], ignore_index=True
        )
        self.scaled_test_df = pd.concat(
            [self.scaled_test_occ_df, self.scaled_test_psa_df], ignore_index=True
        )

        self.train_coords_df = self.train_coords_occ_df.reset_index(drop=True)
        self.val_coords_df = pd.concat(
            [self.val_coords_occ_df, self.val_coords_psa_df], ignore_index=True
        )
        self.test_coords_df = pd.concat(
            [self.test_coords_occ_df, self.test_coords_psa_df], ignore_index=True
        )

    def create_dataset(self,):

        self.__create_occ_df()
        self.__create_psa_df()
        if self.modelling_type == ModellingType.AnomalyDetection:
            self.__create_anomaly_detection_dataset()
        elif self.modelling_type == ModellingType.BinaryClassification:
            self.__create_binary_classification_dataset()

        self.create_vif_dataset()
        self.save()

    def create_vif_dataset(self):
        tempdir = PathUtils.get_temp_dir()
        temp_vif_reference_df_path = tempdir / "temp.csv"
        self.scaled_train_df.to_csv(temp_vif_reference_df_path, index=False)
        vif_calculator = VIFCalculator(
            dataset_path=temp_vif_reference_df_path, output_column="label"
        )
        vif_calculator.calculate_vif()

        self.train_vif_df = self.scaled_train_df[
            vif_calculator.get_optimous_columns_with_label()
        ]
        self.val_vif_df = self.scaled_val_df[
            vif_calculator.get_optimous_columns_with_label()
        ]
        self.test_vif_df = self.scaled_test_df[
            vif_calculator.get_optimous_columns_with_label()
        ]
        self.vif_decision_df = vif_calculator.get_vif_df()

    def save(self):

        # coords df
        self.train_coords_df.to_csv(
            self.species_dataset_path / "train_coords.csv", index=False
        )
        self.val_coords_df.to_csv(
            self.species_dataset_path / "valid_coords.csv", index=False
        )
        self.test_coords_df.to_csv(
            self.species_dataset_path / "test_coords.csv", index=False
        )

        # data df
        self.scaled_train_df.to_csv(
            self.species_dataset_path / "train.csv", index=False
        )
        self.scaled_val_df.to_csv(self.species_dataset_path / "valid.csv", index=False)
        self.scaled_test_df.to_csv(self.species_dataset_path / "test.csv", index=False)

        # vif df
        self.train_vif_df.to_csv(
            self.species_dataset_path / "vif_train.csv", index=False
        )
        self.val_vif_df.to_csv(self.species_dataset_path / "vif_valid.csv", index=False)
        self.test_vif_df.to_csv(self.species_dataset_path / "vif_test.csv", index=False)
        self.vif_decision_df.to_csv(
            self.species_dataset_path / "vif_decision_df.csv", index=False
        )


class DatasetCreationJobPrevious:
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

        self.occ_dataset_builder.build(species_gdf)
        occ_df = self.occ_dataset_builder.get_dataset()
        coords_occ_df = self.occ_dataset_builder.get_coordinates_df()
        scaled_occ_df = self.min_max_scaler.scale_df(occ_df)

        self.psa_dataset_builder = PseudoAbsensesDatasetBuilder(
            root_data_dirpath=self.root_data_dirpath,
            ps_generator_type=self.ps_generator_type,
            scaled_occurrence_df=scaled_occ_df,
            min_max_scaler=self.min_max_scaler,
        )
        number_pseudo_absenses = int(len(occ_df) * self.ps_proportion)
        self.psa_dataset_builder.build(number_pseudo_absenses=number_pseudo_absenses)

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
            df = df.sample(frac=1)
            df_occ = df[df["label"] == 1]
            df_psa = df[df["label"] == 0]
            num_inference_data = int(
                len(df_occ) * self.inference_proportion_from_all_data
            )
            df_train = df_occ[num_inference_data:].reset_index(drop=True)
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
