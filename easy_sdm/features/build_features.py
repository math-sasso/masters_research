import gc
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib import pyplot
from osgeo import gdal
from sklearn.utils import Bunch


class RasterInformationCollector(object):
    """
    This class is reponsable for extracting data from rasters on GBIF occurrence locations

    Attributes
    ----------
    raster_base_configs : str
        Directory to save coverages
    coorection_limit : int
        Limit of iterations to correct no information points
    raster_standards : object
        Raster standards object
    """

    def __init__(self, output_dir:str,raster_utils,utils_methods,coorection_limit:int=10):
        """
        Parameters
        ----------
        raster_base_configs : str
            Directory to save coverages
        coorection_limit : int
            Limit of iterations to correct no information points
        raster_utils : object
            Raster standards object
        """

        self.output_dir = output_dir
        self.raster_utils = raster_utils
        self.coorection_limit = coorection_limit
        self.utils_methods = utils_methods


    def _fill_peristent_no_data_values_with_median_value(self,raster_occurrences_array):
        """ For grids that still with empty value after the board points treatment, this function fill it with the mean value"""

        median_value = np.median(raster_occurrences_array[[raster_occurrences_array!=self.raster_utils.no_data_val]])
        for i,elem in enumerate(raster_occurrences_array):
            if elem == self.raster_utils.no_data_val:
                raster_occurrences_array[i] = median_value
        
        return raster_occurrences_array

    def save_coverges_to_numpy(self,specie_dir:str,species_name:str,root_raster_files_list:List[str]):
        """ Save all extracted to a numpy array"""

        data = gpd.read_file(specie_dir)
        coordinates = np.array((np.array(data['LATITUDE']),np.array(data['LONGITUDE']))).T
            
        # determine coverage values for each of the training & testing points
        Long = coordinates[:,1]
        Lat = coordinates[:,0]

        all_env_values_list = []
        for i,fp in enumerate(root_raster_files_list):
            
            # Exctraction occurences from rasters. As each raster file can have a 
            # different resolution, ix and iy are calculated in every step.
            raster_array,_,xgrid,ygrid,_,_ = self.raster_utils.get_raster_infos(fp)
            ix = np.searchsorted(xgrid,Long)
            iy = np.searchsorted(ygrid,Lat)
            raster_occurrences_array = raster_array[-iy, ix].T
            
            #treating cases where points that should be inside country are outside
            del raster_array

            #tretaing cases that still with no data values
            raster_occurrences_array= self._fill_peristent_no_data_values_with_median_value(raster_occurrences_array)
            
            #selecting the env value on the occurrence position
            all_env_values_list.append(raster_occurrences_array)

            del raster_occurrences_array
            del ix
            del iy
            gc.collect()
            

        coverage= np.stack([value for value in all_env_values_list]).T
        del all_env_values_list
        gc.collect() 

        self.utils_methods.save_nparray_to_folder(coverage,self.output_dir,species_name)
        
        del coverage
        gc.collect()
    