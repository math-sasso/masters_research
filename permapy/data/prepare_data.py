import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
import subprocess
import shutil
from osgeo import ogr, gdal
import rasterio
from rasterio import Affine
from rasterio.plot import show

from permapy.utils import Utils
from permapy.raster_utils import Raster_Utils

# Creating Instances
raster_utils = Raster_Utils()
utils_methods = Utils()



def standarize_raster_to_brazil_region(image_path:str,base_output_root:str,country_limits,utils_methods):
  img_name = image_path.split("/")[-1]
  print(f"Standarizing image: {img_name}...")

  def get_window_from_extent(aff,country_limits):
    """ Get a portion form a raster array based on the country limits"""
    col_start, row_start = ~aff * (country_limits[0],country_limits[3])
    col_stop, row_stop = ~aff * (country_limits[1],country_limits[2])
    return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))


  #Country Limits
  security_limt = 1
  x_min_limit = country_limits[0] -security_limt # DMS Coords: 07°32′39″S 073°59′04″W; Decimal Coords Geohack: -7.544167, -73.984444; Corrdenada Decimal epsg.io: -7.535403, -73.981934
  x_max_limit = country_limits[1] +security_limt # DMS Coords: 07°09′28″S 034°47′38″W; Decimal Coords Geohack: -20.474444, -28.840556; Corrdenada Decimal epsg.io: -7.155017, -34.792929
  y_min_limit = country_limits[2] -security_limt # DMS Coords: 33°45′09″S 053°22′07″W; Decimal Coords Geohack: -33.7525, -53.368611; Corrdenada Decimal epsg.io: -33.750035, -53.407288
  y_max_limit = country_limits[3] +security_limt # DMS Coords: 05°15′05″N 060°12′33″W; Decimal Coords Geohack: 5.251389, -60.209167; Corrdenada Decimal epsg.io: 5.271478, -60.214691  
  country_limits_secure = (x_min_limit,x_max_limit,y_min_limit,y_max_limit)

  #Getting Src Image Infos
  src=rasterio.open(image_path)

  window_region = get_window_from_extent(src.meta['transform'],country_limits_secure)
  band=src.read(1,window =window_region)
  data = band.copy()
  profile = src.profile
  
  #Default Affine Options
  resolution = profile['transform'][0]
  rot1 = profile['transform'][1]
  x_init_point = profile['transform'][2]
  rot2 = profile['transform'][3]
  n_resolution = profile['transform'][4]
  y_init_point = profile['transform'][5]

  #Creating Grids
  xgrid = np.arange(x_min_limit, x_max_limit, resolution)
  ygrid = np.arange(y_min_limit, y_max_limit, resolution)
  Nx = len(xgrid)
  Ny = len(ygrid)

  #1 - Converting data to np.float32
  if profile['dtype'] != np.float32:
      # data = data.astype(rasterio.float32)
      data = np.float32(data)
      profile['dtype'] = np.float32
  
  #2 - Converting no data to -9999.0
  # Note that form brazilian mask. The background is 0. So it wont be affected
  if profile['nodata'] != -9999.0:
      data = np.where(data < -9999.0 ,-9999.0, data)
      profile['nodata'] =  -9999.0

  #3 - Changing width and height to the cropped region  
  profile['width'] = data.shape[1]
  profile['height'] = data.shape[0]

  #4 - setting CRS
  profile['crs'] = {'init': 'EPSG:4326'}

  #5 - Changing Affine parameters
  # Example: Affine(0.008333333333333333, 0.0, -180.0, 0.0, -0.008333333333333333, 90.0)
  profile["transform"] =  Affine(resolution,rot1,x_min_limit,rot2,n_resolution,y_max_limit)

  #6 - Saving Destination Pathj
  objective_root_name = "/" + base_output_root.split("/")[-1] + "/"
  default_root_name = "/" + image_path.split("/")[-3] + "/"
  dst_path = image_path.replace(default_root_name,objective_root_name)
  dst_root_folder = "/".join(dst_path.split("/")[:-1])
  utils_methods.create_folder_structure(dst_root_folder)

  print(f"Destination Profile:\n{profile}")
  print(f"Destination Array:\n{data}")
  print("----------------------------------")
  with rasterio.open(dst_path, 'w', **profile) as dst:
      dst.write(data, 1)
      