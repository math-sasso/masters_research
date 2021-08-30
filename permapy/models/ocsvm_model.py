import os
import numpy as np
import geopandas as gpd
from typing import List,Tuple,Dict
from sklearn import svm, metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import rasterio
import matplotlib.pyplot as plt
import pandas as pd


class SDM:
    #OneClassSVMModel

  
  """
  This class is reponsable for performing fits and predictions for the species ditribution problem using OneClassSVM 
   Attributes
  ----------
  nu : float
      An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1]. By default 0.5 will be taken.
  kenel : str
      Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.
  gamma : object
      Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
      if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
      if ‘auto’, uses 1 / n_features.
  seed : object
    Aleatory seed
  raster_utils : object
    Raster standards object
  utils_methods : object
    Utils object
  land_reference : array
    Array used as the land reference
  
  """
  def __init__(self,hyperparams:Dict,raster_utils, utils_methods,land_reference_path:str,stacked_rasters_path:str,brazil_vars_mean_std_path:str,output_base_folder:str):
    """    
    Parameters
    ----------
    hyperparams : Dict
        Set of hyperparameters for the model(nu,kernel,gamma,seed)
    raster_utils : Object
        Raster standards object
    utils_methods : Object
        Utils object
    land_reference_path : str
        Path to a raster used as land refence
       
    """
  
    #-------------- hyperparams
    self.nu = hyperparams["nu"]
    self.kernel = hyperparams["kernel"]
    self.gamma = hyperparams["gamma"]
    self.seed = hyperparams["seed"]
    
    #-------------- Auxiliary Classes
    self.raster_utils = raster_utils
    self.utils_methods = utils_methods

    #------------- Useful Information
    self.output_base_folder = output_base_folder
    self.land_reference,_,_,_,_,_= self.raster_utils.get_raster_infos(land_reference_path)
    np.random.seed(self.seed)
    self.stacked_raster_coverages = utils_methods.retrieve_data_from_np_array(stacked_rasters_path) 
    brazil_vars_mean_std_df = pd.read_csv(brazil_vars_mean_std_path)
    self.mean_vars = np.float32(brazil_vars_mean_std_df['mean'].to_numpy())
    self.std_vars = np.float32(brazil_vars_mean_std_df['std'].to_numpy())
    
    # Extracting coverages land
    idx = np.where(self.land_reference == self.raster_utils.positive_mask_val) # Coords X and Y in two tuples where condition matchs (array(),array())
    self.idx_X = idx[0]
    self.idx_Y = idx[1]
    # self.utils_methods.save_nparray_to_folder(self.idx_X,self.output_base_folder,"Idx_X_Brazilian_Territory")
    # self.utils_methods.save_nparray_to_folder(self.idx_Y,self.output_base_folder,"Idx_Y_Brazilian_Territory")


  def fit(self,species_bunch):
    """ Fitting data with normalized data """

    train_cover_std = (species_bunch['raster_data_train'] - self.mean_vars) / self.std_vars
    train_cover_std[np.isnan(train_cover_std)] = 0 #Nan values comes from std=0 in some variable
    clf = svm.OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
    clf.fit(train_cover_std)
    return clf
  
  def predict_land(self,clf):
    """ Predict adaptability for every valid point on the map """

    stacked_raster_coverages_shape = self.stacked_raster_coverages.shape
    print('Shape stacked_raster_coverages: ',stacked_raster_coverages_shape)
    
    #Performing Predictions
    raster_coverages_land = self.stacked_raster_coverages[:, self.idx_X, self.idx_Y].T
    for k in range(raster_coverages_land.shape[1]):
      raster_coverages_land[:,k][raster_coverages_land[:,k]<=self.raster_utils.no_data_val] = self.mean_vars[k]
    
    scaled_coverages_land = (raster_coverages_land - self.mean_vars) / self.std_vars
    del raster_coverages_land

    scaled_coverages_land[np.isnan(scaled_coverages_land)] = 0
    global_pred = clf.decision_function(scaled_coverages_land)
    del scaled_coverages_land

    #Setting Spatial Predictions
    Z = np.ones((stacked_raster_coverages_shape[1], stacked_raster_coverages_shape[2]), dtype=np.float32)
    # Z *= global_pred.min()
    # Z *=-1 #This will be necessary to set points outside map to the minimum
    Z*= self.raster_utils.no_data_val #This will be necessary to set points outside map to the minimum
    Z[self.idx_X, self.idx_Y] = global_pred

    del global_pred

    #Setting no data values
    Z[self.land_reference == self.raster_utils.no_data_val] = self.raster_utils.no_data_val

    return Z

  def predict_test_occurences(self,species_bunch,clf):
    """ Fitting adaptability only for test set data """

    scaled_species_raster_test = (species_bunch['raster_data_test'] - self.mean_vars) / self.std_vars
    scaled_species_raster_test[np.isnan(scaled_species_raster_test)] = 0
    pred_test = clf.decision_function(scaled_species_raster_test)
    return pred_test



  def perform_K_folder_preidction(self,species_occurence_path:str,specie_shp_path:str,list_raster_files:List,K:int):
    """ Perform K times the prediction pipeline """
    
    #1 Getting species name
    species_name = species_occurence_path.split("/")[-1].split(".")[0]
    print(f">>>>>>>>>> Performing Kofld prediction for {species_name} <<<<<<<<<<")

    #2 Recovering occurrences data
    species_gdf = gpd.read_file(specie_shp_path)
    coordinates = np.array((np.array(species_gdf['LATITUDE']),np.array(species_gdf['LONGITUDE']))).T 
    species_raster_data = self.utils_methods.retrieve_data_from_np_array(species_occurence_path)

    #3 reating kfolds object
    kf = KFold(n_splits=K,random_state=self.seed, shuffle=True)

    #4 Executing Pipeline
    for i, (train_index, test_index) in enumerate(kf.split(species_raster_data)):
      print(f"------------------------------ KFold {i+1} ------------------------------")
      #creating Kfold Folder Structure
      kfold_path = os.path.join(self.output_base_folder,species_name,f"KFold{i+1}")
      self.utils_methods.create_folder_structure(kfold_path)


      species_raster_data_train, species_raster_data_test = species_raster_data[train_index], species_raster_data[test_index]
      coords_train, coords_test = coordinates[train_index], coordinates[test_index]
      species_bunch = {'species_name':species_name,
                       'raster_data_train':species_raster_data_train,
                       'raster_data_test':species_raster_data_test,
                       'coords_train':coords_train,
                       'coords_test':coords_test}
      
      clf = self.fit(species_bunch)

      #predicting values only for test points
      pred_test = self.predict_test_occurences(species_bunch,clf)

      #predicting land values
      Z = self.predict_land(clf)

      #save Z
      self.utils_methods.save_nparray_to_folder(Z,kfold_path,"Land_Prediction")
      del Z
      #save pred_test
      self.utils_methods.save_nparray_to_folder(pred_test,kfold_path,"Test_Prediction")
      del pred_test
      #save coords_train
      self.utils_methods.save_nparray_to_folder(species_bunch['coords_train'],kfold_path,"Coords_Train")
      #save_coords_test
      self.utils_methods.save_nparray_to_folder(species_bunch['coords_test'],kfold_path,"Coords_Test")
      #raster_data_train
      self.utils_methods.save_nparray_to_folder(species_bunch['raster_data_train'],kfold_path,"Species_Raster_Data_Train")
      #raster_data_test
      self.utils_methods.save_nparray_to_folder(species_bunch['raster_data_test'],kfold_path,"Species_Raster_Data_Test")
      del  species_bunch