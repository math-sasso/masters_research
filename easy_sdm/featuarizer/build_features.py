import gc
from typing import List, Tuple
from easy_sdm.data.species_data import SpeciesInfoExtractor

import geopandas as gpd
import numpy as np
import rasterio
from configs import configs
from data.raster_data import RasterInfoExtractor
from data.data_loader import RasterLoader
from featuarizer.pseudo_species_generators import BasePseudoSpeciesGenerator,RandomPseudoSpeciesGenerator,RSEPPseudoSpeciesGenerator
from pathlib import Path
from typing import List]

class DatasetBuilder:
    def __init__(self,raster_path_list:List[Path],ps_generator:BasePseudoSpeciesGenerator) -> None:
        self.ps_generator = ps_generator
        self.

    def build(self):
        pass

class PseudoAbsensesDatasetBuilder():
    def __init__(self):
        pass

    def build(self,ps_generator:BasePseudoSpeciesGenerator):
        pass

class OccurrancesDatasetBuilder():
    def __init__(self,raster_path_list:List[Path]):
        self.raster_path_list = raster_path_list
        self.species_env_extractor = SpeciesEnveriomentExtractor()

    def build(
        self, species_gdf: gpd.GeoDataFrame,
    ):
        """Save all extracted to a numpy array"""

        all_env_values_list = []
        self.species_env_extractor()

        data = gpd.read_file(specie_dir)

        coverage = np.stack([value for value in all_env_values_list]).T
        del all_env_values_list
        gc.collect()

        self.utils_methods.save_nparray_to_folder(
            coverage, self.output_dir, species_name
        )

        del coverage
        gc.collect()

class SpeciesEnveriomentExtractor(object):
    def __init__(self):
        self.configs = configs

        # Species related arguments
        self.__species_info_extractor = None
        self.__species_geodataframe = None

        # Envs related arguments
        self.__raster_info_extractor = None
        self.__raster = None

    def set_env_layer(self,raster:rasterio.io.DatasetReader):
        self.__raster_info_extractor = RasterInfoExtractor(raster)
        self.__raster = raster

    def set_species(self,species_geodataframe:gpd.GeoDataFrame):
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
                [raster_occurrences_array != self.configs['maps']['no_data_val']]
            ]
        )
        for i, pixel_value in enumerate(raster_occurrences_array):
            if pixel_value == self.configs['maps']['no_data_val']:
                raster_occurrences_array[i] = median_value

        return raster_occurrences_array

    def __approximate_no_data_pixels(self,raster_occurrences_array:np.ndarray):
        """[Repair problems on points very near to the the country boarders make the point change position in the raster center direction]

        Args:
            raster_occurrences_array (np.ndarray): [description]

        Raises:
            KeyError: [description]

        Returns:
            [type]: [description]
        """


        raster_array = self.__raster_info_extractor.get_array()
        species_longitudes = self.__species_info_extractor.get_longitudes()
        species_latitudes = self.__species_info_extractor.get_latitudes()

        ix = self.__calc_sepcies_pos_x()
        iy = self.__calc_sepcies_pos_y()

        for i,(pixel_value,long,lat) in enumerate(zip(raster_occurrences_array,species_longitudes,species_latitudes)):

            if pixel_value == self.configs['maps']['no_data_val']:
                incx,incy = 0,0
                k = 0
                while (pixel_value==self.configs['maps']['no_data_val']):
                    if k==100:
                        raise KeyError("Probably there is problem in the raster once it could not find a valid value")
                    #walking coodinates in center map
                    if long >= self.__raster_info_extractor.get_xcenter() and  lat >= self.__raster_info_extractor.get_ycenter():
                        long -= self.raster_standars.resolution
                        lat -= self.raster_standars.resolution
                        incx -= 1
                        incy -= 1
                    if long >= self.__raster_info_extractor.get_xcenter() and  lat <= self.__raster_info_extractor.get_ycenter():
                        long -= self.raster_standars.resolution
                        lat += self.raster_standars.resolution
                        incx -= 1
                        incy += 1
                    if long <= self.__raster_info_extractor.get_xcenter() and  lat <= self.__raster_info_extractor.get_ycenter():
                        long += self.raster_standars.resolution
                        lat += self.raster_standars.resolution
                        incx += 1
                        incy += 1
                    if long <= self.__raster_info_extractor.get_xcenter() and  lat >= self.__raster_info_extractor.get_ycenter():
                        long += self.raster_standars.resolution
                        lat -= self.raster_standars.resolution
                        incx += 1
                        incy -= 1

                    new_ix = ix[i]+incx
                    new_iy = iy[i]+incy

                    pixel_value = raster_array[-new_iy,new_ix].T

                    if pixel_value != self.configs['maps']['no_data_val']:
                        raster_occurrences_array[i] = pixel_value

                    k+=1

        return raster_occurrences_array

    def __calc_sepcies_pos_x(self):
        return np.searchsorted(self.__raster_info_extractor.get_xgrid(), self.__species_info_extractor.__get_longitudes())

    def __calc_sepcies_pos_y(self):
        return np.searchsorted(self.__raster_info_extractor.get_ygrid(), self.__species_info_extractor.__get_latitudes())


    def extract(self):

        # Check if required fields were set
        self.__existance_verifier()

        # Extract enverionment values for coordinates
        ix = self.__calc_sepcies_pos_x()
        iy = self.__calc_sepcies_pos_y()
        raster_array = self.__raster_info_extractor.get_array()
        raster_occurrences_array = raster_array[-iy, ix].T
        del raster_array
        del ix
        del iy

        # Treat with no data values
        raster_occurrences_array = self.__approximate_no_data_pixels(raster_occurrences_array)
        gc.collect()

        # Reset vals for the next call
        self.__reset_vals()

        return raster_occurrences_array