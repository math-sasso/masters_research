from typing import List,Tuple
import numpy as np
import os
import rasterio

class Raster_Utils:
  
  """
  This class is reponsable for define all the raster base parametes used on the project and for applying operations on raster arrays

   Attributes
  ----------
  security_limt : int
    Limit to extrapolate territory boarders in all directions
  x_min_limit : float
      Country most western point considering security_limit
  x_max_limit : float
      Country most eastern point considering security_limit
  y_min_limit : float
      Country most south point considering security_limit
  y_max_limit : float
      Country most north point considering security_limit
  resolution : int
      Raster resolution
  crs : int
      CRS code for map projection
  no_data_val : int
      No data value
  positive_mask_val : int
      Positive mask value
  negative_mask_val : int
      Negative mask value
  xgrid : array
      Grid of values for refernce map on X
  ygrid : array
      Grid of values for refernce map on Y
  x_center_point : int
      Center point X coordinate
  y_center_point : int
      Center point Y coordinate
  """

  def __init__(self):
      """
      Parameters
      ----------
      raster_base_configs : Dict
          Configurations for raster Standards (resolution,crs,no_data_val,positive_mask_val,negative_mask_val and country_limits)
      """

      self.security_limt = 1
      country_limits = (-73.981934,-34.792929, -33.750035, 5.271478)
      self.x_min_limit = country_limits[0] -self.security_limt # Coordenadas DMS: 07°32′39″S 073°59′04″W; Coordenada Decimal Geohack: -7.544167, -73.984444; Corrdenada Decimal epsg.io: -7.535403, -73.981934
      self.x_max_limit = country_limits[1] +self.security_limt # Coordenadas DMS: 07°09′28″S 034°47′38″W; Coordenada Decimal Geohack: -20.474444, -28.840556; Corrdenada Decimal epsg.io: -7.155017, -34.792929
      self.y_min_limit = country_limits[2] -self.security_limt # Coordenadas DMS: 33°45′09″S 053°22′07″W; Coordenada Decimal Geohack: -33.7525, -53.368611; Corrdenada Decimal epsg.io: -33.750035, -53.407288
      self.y_max_limit = country_limits[3] +self.security_limt # Coordenadas DMS: 05°15′05″N 060°12′33″W; Coordenada Decimal Geohack: 5.251389, -60.209167; Corrdenada Decimal epsg.io: 5.271478, -60.214691
      self.crs = 4326
      self.no_data_val = -9999.0
      self.positive_mask_val = 255
      self.negative_mask_val = 0

  def _read_and_check(self,raster,raster_name):
      """ Performs verifications and standarizations for raster arrays """

      print(f"Reading raster {raster_name}")

      #1 Check if orientaion from Affine position 4 is negative
      res_n_s = raster.meta['transform'][4]
      if res_n_s > 0:
        raise Exception("Behavior not expected. The North South resolution is excpected to be negative")

      #2 Checking the number of raster layers 
      if raster.meta['count']>1:
        raise Exception("For some reason there are more than one layer in this raster")
      if raster.meta['count']==0:
        raise Exception("For some reason this raster is empty")

      #3 Checking CRS
      raster_code = int(raster.crs.data['init'].split(':')[1])
      if raster_code != self.crs:
        raise Exception("Sorry,crs from this raster is no EPSG:4326")
        # raster = raster.to_crs(epsg=self.crs)

      #4 converting nodata value if necessary
      if raster.nodata != self.no_data_val:
        raise Exception(f"For some reason array no data val is not {self.no_data_val}")
        
      #5 Asserting that numpy array will be float32
      if raster.meta['dtype'] != 'float32': 
        raise Exception(f"For some reason array dtype is not float32")
      
      #6 Extracting only the window from Raster Standars Object
      raster_array = raster.read(1)

      #7 Setting raster to none. The information that matters is the rater aray
      raster = None
      
      return raster_array
  
  def get_raster_infos(self,raster_path):
      """ Returns infos (raster_array,xgrid,ygrid) from a contry mask reference array"""
      
      # While required because of an occasional error. Sometimes happens, sometimes not.
      check_read = 0
      while check_read == 0:
        try:
          raster = rasterio.open(raster_path)
          check_read = 1
        except rasterio.errors.RasterioIOError:
          check_read = 0
	                
      resolution = raster.meta['transform'][0]
      raster_array = self._read_and_check(raster,raster_path.split("/")[-1])
      xgrid = np.arange(self.x_min_limit, self.x_min_limit+raster_array.shape[1]*resolution, resolution)
      ygrid = np.arange(self.y_min_limit, self.y_min_limit+raster_array.shape[0]*resolution, resolution)
      x_center = np.mean(xgrid)
      y_center = np.mean(ygrid)
      return raster_array,raster.meta,xgrid,ygrid,x_center,y_center
  
