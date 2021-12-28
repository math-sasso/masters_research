import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from easy_sdm.configs import configs
from easy_sdm.data import (RasterInfoExtractor, RasterLoader,
                           SpeciesInfoExtractor)

class BasePseudoSpeciesGenerator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate(self):
        pass


class RandomPseudoSpeciesGenerator(BasePseudoSpeciesGenerator):
    def __init__(self) -> None:
        raise NotImplementedError("This class is not yet implemented")

    def generate(self):
        pass


class RSEPPseudoSpeciesGenerator(BasePseudoSpeciesGenerator):
    def __init__(self) -> None:
        raise NotImplementedError("This class is not yet implemented")

    def generate(self):
        pass

class SpeciesEnveriomentExtractor:
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

    def __fill_peristent_no_data_values_with_median_value(
        self, raster_occurrences_array
    ):
        """For grids that still with empty value after the board points treatment, this function fill it with the mean value"""

        median_value = np.median(
            raster_occurrences_array[
                [raster_occurrences_array != self.configs["maps"]["no_data_val"]]
            ]
        )
        for i, pixel_value in enumerate(raster_occurrences_array):
            if pixel_value == self.configs["maps"]["no_data_val"]:
                raster_occurrences_array[i] = median_value

        return raster_occurrences_array

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

        for i, (pixel_value, long, lat) in enumerate(
            zip(raster_occurrences_array, species_longitudes, species_latitudes)
        ):

            if pixel_value == self.configs["maps"]["no_data_val"]:
                incx, incy = 0, 0
                k = 0
                while pixel_value == self.configs["maps"]["no_data_val"]:
                    if k == 100:
                        raise KeyError(
                            "Probably there is problem in the raster once it could not find a valid value"
                        )
                    # walking coodinates in center map
                    if (
                        long >= self.__raster_info_extractor.get_xcenter()
                        and lat >= self.__raster_info_extractor.get_ycenter()
                    ):
                        long -= resolution
                        lat -= resolution
                        incx -= 1
                        incy -= 1
                    if (
                        long >= self.__raster_info_extractor.get_xcenter()
                        and lat <= self.__raster_info_extractor.get_ycenter()
                    ):
                        long -= resolution
                        lat += resolution
                        incx -= 1
                        incy += 1
                    if (
                        long <= self.__raster_info_extractor.get_xcenter()
                        and lat <= self.__raster_info_extractor.get_ycenter()
                    ):
                        long += resolution
                        lat += resolution
                        incx += 1
                        incy += 1
                    if (
                        long <= self.__raster_info_extractor.get_xcenter()
                        and lat >= self.__raster_info_extractor.get_ycenter()
                    ):
                        long += resolution
                        lat -= resolution
                        incx += 1
                        incy -= 1

                    new_ix = ix[i] + incx
                    new_iy = iy[i] + incy

                    pixel_value = raster_array[-new_iy, new_ix].T

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
        raster_occurrences_array = raster_array[-iy, ix].T
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


class EnverionmentLayersStacker:
    def __init__(self, raster_path_list: List[Path]) -> None:
        self.raster_path_list = raster_path_list

    def stack_and_save(self, output_path: Path):
        if not str(output_path).endswith(".npy"):
            raise TypeError("output_path must ends with .npy")

        coverage = self.stack()
        with open(output_path, "wb") as f:
            np.save(f, coverage)

        del coverage
        gc.collect()

    def stack(self):

        all_env_values_list = []
        for path in self.raster_path_list:
            raster = RasterLoader(path).load_dataset()
            raster_array = RasterInfoExtractor(raster).get_array()
            all_env_values_list.append(raster_array)
            del raster
            del raster_array
            gc.collect()

        coverage = np.stack(all_env_values_list)

        return coverage


class DatasetCreator:
    def __init__(
        self, raster_path_list: List[Path], ps_generator: BasePseudoSpeciesGenerator
    ) -> None:
        self.ps_generator = ps_generator

    def create_dataset(self):
        pass


class BaseDatasetBuilder(ABC):
    def __init__(self, raster_path_list: List[Path]):
        self.raster_path_list = raster_path_list

    @abstractmethod
    def build(self):
        pass

    def __get_var_names(self):
        return [path.name.split(".")[0] for path in self.raster_path_list]

    def create_df(
        self,
        all_env_values_species_list: List[np.ndarray],
        coordinates: List[np.ndarray],
    ):

        env_matrix = np.vstack(all_env_values_species_list)
        complete_array = np.hstack([env_matrix.T, coordinates])
        del env_matrix
        del coordinates

        columns = ["lat", "lon"] + self.__get_var_names()
        df = pd.DataFrame(complete_array, columns=columns)
        df = df.set_index(["lat", "lon"])

        del complete_array
        gc.collect()

        return df


class PseudoAbsensesDatasetBuilder(BaseDatasetBuilder):
    def __init__(
        self, raster_path_list: List[Path], ps_generator: BasePseudoSpeciesGenerator
    ):
        super().__init__(raster_path_list)
        pass

    def build(self):
        pass


class OccurrancesDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, raster_path_list: List[Path]):
        super().__init__(raster_path_list)
        self.species_env_extractor = SpeciesEnveriomentExtractor()

    def build(
        self,
        species_gdf: gpd.GeoDataFrame,
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

        df = self.create_df(all_env_values_species_list, coordinates)
        df["label"] = 1
        return df
