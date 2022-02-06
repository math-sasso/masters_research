import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from easy_sdm.configs import configs
from easy_sdm.data import RasterInfoExtractor, RasterLoader, SpeciesInfoExtractor
from easy_sdm.enums import PseudoSpeciesGeneratorType

from .environment_builder import EnverionmentLayersStacker
from .pseudo_species_generators import (
    BasePseudoSpeciesGenerator,
    RSEPPseudoSpeciesGenerator,
)
from .scaler import MinMaxScalerWrapper


class SpeciesEnveriomentExtractor:
    """[This class extracts species information trought a set of raster layers that represent enverionmental conditions]"""

    def __init__(self):
        self.configs = configs

        # Species related arguments
        self.__species_info_extractor = None
        self.__species_geodataframe = None

        # Envs related arguments
        self.__raster_info_extractor = None
        self.__raster = None

    def set_env_layer(self, raster: rasterio.io.DatasetReader):
        self.__raster_info_extractor = RasterInfoExtractor(raster)
        self.__raster = raster

    def set_species(self, species_geodataframe: gpd.GeoDataFrame):
        self.__species_info_extractor = SpeciesInfoExtractor(species_geodataframe)
        self.__species_geodataframe = species_geodataframe

    def __reset_vals(self):
        self.__species_info_extractor = None
        self.__species_geodataframe = None
        self.__raster_info_extractor = None
        self.__raster = None

    def __existance_verifier(self):
        if self.__species_geodataframe is None:
            raise ValueError("Call set_species before extract")
        if self.__raster is None:
            raise ValueError("Call set_env_layer before extract")

    def __approximate_no_data_pixels(self, raster_occurrences_array: np.ndarray):
        """[Repair problems on points very near to the the country boarders make the point change position in the raster center direction]

        Args:
            raster_occurrences_array (np.ndarray): [description]

        Raises:
            KeyError: [description]

        Returns:
            [type]: [description]
        """

        raster_array = self.__raster_info_extractor.get_array()
        resolution = self.__raster_info_extractor.get_resolution()

        species_longitudes = self.__species_info_extractor.get_longitudes()
        species_latitudes = self.__species_info_extractor.get_latitudes()

        ix = self.__calc_species_pos_x()
        iy = self.__calc_species_pos_y()

        # SpeciesInRasterPlotter.plot_all_points(raster_array,ix,iy)

        for i, (pixel_value, long, lat) in enumerate(
            zip(raster_occurrences_array, species_longitudes, species_latitudes)
        ):

            state = "diagonal"

            if pixel_value == self.configs["maps"]["no_data_val"]:
                incx, incy = 0, 0
                k = 0
                while pixel_value == self.configs["maps"]["no_data_val"]:

                    if k % 50 == 0:
                        if state == "diagonal":
                            state = "horizontal"
                        elif state == "horizontal":
                            state = "vertical"
                        elif state == "vertical":
                            state == "diagonal"

                    if k == 200:
                        raise KeyError(
                            "Probably there is problem in the raster once it could not find a valid value"
                        )
                    # walking coodinates in center map
                    if state == "diagonal":
                        if (
                            long >= self.__raster_info_extractor.get_xcenter()
                            and lat >= self.__raster_info_extractor.get_ycenter()
                        ):
                            long -= resolution
                            lat -= resolution
                            incx -= 1
                            incy -= 1
                        elif (
                            long >= self.__raster_info_extractor.get_xcenter()
                            and lat <= self.__raster_info_extractor.get_ycenter()
                        ):
                            long -= resolution
                            lat += resolution
                            incx -= 1
                            incy += 1
                        elif (
                            long <= self.__raster_info_extractor.get_xcenter()
                            and lat <= self.__raster_info_extractor.get_ycenter()
                        ):
                            long += resolution
                            lat += resolution
                            incx += 1
                            incy += 1
                        elif (
                            long <= self.__raster_info_extractor.get_xcenter()
                            and lat >= self.__raster_info_extractor.get_ycenter()
                        ):
                            long += resolution
                            lat -= resolution
                            incx += 1
                            incy -= 1
                    elif state == "horizontal":
                        if long <= self.__raster_info_extractor.get_xcenter():
                            long += resolution
                            incx += 1
                        else:
                            long -= resolution
                            incx -= 1
                    elif state == "vertical":
                        if lat <= self.__raster_info_extractor.get_ycenter():
                            lat += resolution
                            incy += 1
                        else:
                            lat -= resolution
                            incy -= 1

                    newx_point = ix[i] + incx
                    newy_point = iy[i] + incy

                    pixel_value = raster_array[-newy_point, newx_point]

                    if pixel_value != self.configs["maps"]["no_data_val"]:
                        raster_occurrences_array[i] = pixel_value

                    k += 1

        return raster_occurrences_array

    def __calc_species_pos_x(self):
        return np.searchsorted(
            self.__raster_info_extractor.get_xgrid(),
            self.__species_info_extractor.get_longitudes(),
        )

    def __calc_species_pos_y(self):
        return np.searchsorted(
            self.__raster_info_extractor.get_ygrid(),
            self.__species_info_extractor.get_latitudes(),
        )

    def extract(self):

        # Check if required fields were set
        self.__existance_verifier()

        # Extract enverionment values for coordinates
        ix = self.__calc_species_pos_x()
        iy = self.__calc_species_pos_y()
        raster_array = self.__raster_info_extractor.get_array()
        raster_occurrences_array = raster_array[-iy, ix]
        del raster_array
        del ix
        del iy

        # Treat with no data values
        raster_occurrences_array = self.__approximate_no_data_pixels(
            raster_occurrences_array
        )
        gc.collect()

        # Reset vals for the next call
        self.__reset_vals()

        return raster_occurrences_array


class OccurrancesDatasetBuilder:
    def __init__(self, raster_path_list: List[Path]):
        self.raster_path_list = raster_path_list
        self.species_env_extractor = SpeciesEnveriomentExtractor()

    def __get_var_names(self):
        return [path.name.split(".")[0] for path in self.raster_path_list]

    def __create_df(
        self,
        all_env_values_species_list: List[np.ndarray],
        coordinates: List[np.ndarray],
    ):
        env_matrix = np.vstack(all_env_values_species_list)
        complete_array = np.hstack([coordinates, env_matrix.T])
        del env_matrix
        del coordinates

        columns = ["lat", "lon"] + self.__get_var_names()
        df = pd.DataFrame(complete_array, columns=columns)
        df = df.set_index(["lat", "lon"])

        del complete_array
        gc.collect()

        return df

    def build(
        self, species_gdf: gpd.GeoDataFrame,
    ):
        """Save all extracted to a numpy array"""

        all_env_values_species_list = []
        coordinates = SpeciesInfoExtractor(species_gdf).get_coordinates()
        for path in self.raster_path_list:
            raster = RasterLoader(path).load_dataset()

            self.species_env_extractor.set_env_layer(raster)
            self.species_env_extractor.set_species(species_gdf)
            raster_occurrences_array = self.species_env_extractor.extract()
            all_env_values_species_list.append(raster_occurrences_array)
            del raster
            del raster_occurrences_array
            gc.collect()

        df = self.__create_df(all_env_values_species_list, coordinates)
        df["label"] = 1
        return df


class PseudoAbsensesDatasetBuilder:
    def __init__(
        self, ps_generator_type: PseudoSpeciesGeneratorType,
    ):
        self.ps_generator_type = ps_generator_type

        self.__define_ps_generator()

    def __define_ps_generator(self):

        region_mask_raster = rasterio.open(
            Path.cwd() / "data/processed_rasters/others/brazilian_mask.tif"
        )

        if self.ps_generator_type is PseudoSpeciesGeneratorType.RSEP:
            hyperparameters = configs["pseudo_species"]["RSEP"]
            stacked_raster_coverages = EnverionmentLayersStacker().load(
                Path.cwd() / "data/numpy/env_stack.npy"
            )
            ps_generator = RSEPPseudoSpeciesGenerator(
                hyperparameters=hyperparameters,
                region_mask_raster=region_mask_raster,
                stacked_raster_coverages=stacked_raster_coverages,
            )
        elif self.ps_generator_type is PseudoSpeciesGeneratorType.Random:
            raise NotImplementedError()

        self.ps_generator = ps_generator

    def build(self, occurrence_df: pd.DataFrame, number_pseudo_absenses: int):
        self.ps_generator.fit(occurrence_df)
        pseudo_absenses_df = self.ps_generator.generate(number_pseudo_absenses)
        return pseudo_absenses_df


class SDMDatasetCreator:
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
