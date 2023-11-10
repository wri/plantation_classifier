#!/usr/bin/env python

from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio as rs
import pickle
import os
import numpy as np
import pandas as pd
import copy
from pyproj import Proj, transform
import math

def cm_roc_pr(model, y_test, pred, probs_pos):

    ''' 
    Produces a confusion matrix, ROC curve and precision recall curve
    '''
    
    with open(f'../models/{model}.pkl', 'rb') as file:  
        model = pickle.load(file)

     # Confusion Matrix
    cm = confusion_matrix(y_test, pred, labels=model.classes_)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot();

    # ROC AUC and Precision Recall Curves
    plt.figure(figsize=(17,6)) 
    
    plt.subplot(1,2,1)
    
    # calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probs_pos)

    # plot roc curve and no skill model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=model, color='green')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right');
    
    plt.subplot(1,2,2)
    
    # calculate precision-recall curve
    fpr, tpr, thresholds = precision_recall_curve(y_test, probs_pos)

    # plot roc curve and no skill model
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0,1], [no_skill, no_skill], linestyle='--', label='No Skill')

    plt.plot(fpr, tpr, marker='.', label=model, color='purple')
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left');
    
    return None


def roc_curve_comp(X_test, y_test, model_names):

    '''
    Plots the ROC Curve for all listed models,
    '''
    
    plt.figure(figsize=(17,6)) 
    
    # ROC curve
    for m in model_names:
        
        with open(f'../models/{m}.pkl', 'rb') as file:  
             model = pickle.load(file)

        plt.subplot(1,2,1)
        
        # calculate and plot ROC curve
        probs = model.predict_proba(X_test)
        probs_pos = probs[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, probs_pos)
        plt.plot(fpr, tpr, marker=',', label=m)
    
    # plot no skill and custom settings
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right');
    
    # AUC curve
    for m in model_names:
        
        with open(f'../models/{m}.pkl', 'rb') as file:  
             model = pickle.load(file)

        plt.subplot(1,2,2)

        # calculate and plot precision-recall curve
        probs = model.predict_proba(X_test)
        probs_pos = probs[:, 1]
        fpr, tpr, thresholds = precision_recall_curve(y_test, probs_pos)
        plt.plot(fpr, tpr, marker=',', label=m)
    
    # plot no skill and custom settings
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0,1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.4, 1.05])
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left');
    
    return None


def learning_curve_comp(model_names, X_train, y_train, x_max):

    '''
    Plots a learning curve to visualize the performance of given machine
    learning models by comparing their training and testing scores as the amount of training
    data increases. It uses k-fold cross-validation to estimate the model performance.
    This also allows for model comparison.

    '''

    plt.figure(figsize = (13,6))
    
    colors = ['royalblue',
              'maroon', 
              'magenta', 
              'gold', 
              'limegreen'] 

    for i, x in zip(model_names, colors[:len(model_names)+1]):

        filename = f'../models/{i}.pkl'

        with open(filename, 'rb') as file:
            model = pickle.load(file)

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, 
                                                                              X_train, 
                                                                              y_train, 
                                                                              cv=5, 
                                                                              return_times=True,
                                                                              verbose=0) 

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.grid()
        plt.plot(train_sizes, train_scores_mean, "x-", color=x, label=f"{i[0:4]} Train")
        plt.plot(train_sizes, test_scores_mean, ".-", color=x, label=f"{i[0:4]} Test")
    
    plt.xlim([1000, x_max])
    plt.ylim([0.0, 1.2])
    plt.title(f'Learning Curve Comparison for {len(model_names)} Models')
    plt.xlabel('Training Samples')
    plt.ylabel('Score')
    plt.legend(title='Models', loc='lower right');        
        
    return None


## in progress
def visualize_plotpreds(model_name, v_train_data, X_test):
    
    filename = f'../models/{model_name}_model_{v_train_data}.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    preds = model.predict(X_test)
 
    sns.heatmap(preds.reshape((14,14)), vmin=0, vmax=.8).set_title(model_name)

    return None

def make_bbox(country: str, tile_idx: tuple, expansion: int = 10) -> list:
    """
    Makes a (min_x, min_y, max_x, max_y) bounding box that
    is 2 * expansion 300 x 300 meter ESA LULC pixels. 

       Parameters:
            initial_bbx (list): [min_x, min_y, max_x, max_y]
            expansion (int): 1/2 number of 300m pixels to expand

       Returns:
            bbx (list): expanded [min_x, min_y, max_x, max_y]
    """
    bbx_df = pd.read_csv(f"data/{country}.csv", engine="pyarrow")

    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
    
    # extract the XY of interest as a dataframe
    bbx_df = bbx_df[bbx_df['X_tile'] == int(x)]
    bbx_df = bbx_df[bbx_df['Y_tile'] == int(y)]
    bbx_df = bbx_df.reset_index(drop = True)

    # creates a point [min x, min y, max x, max y] (min and max will be the same)
    initial_bbx = [bbx_df['X'][0], bbx_df['Y'][0], bbx_df['X'][0], bbx_df['Y'][0]]
    
    # starting at a centroid, want to create a 6x6km box
    # pixels match up with ESA area adjusted/locally sized pixels 
    # 10*300m pixels to the left (~6km) and to the right (~6km)
    
    multiplier = 1/360 # Sentinel-2 pixel size in decimal degrees
    bbx = copy.deepcopy(initial_bbx)
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    
    # return the dataframe and the array
    return bbx


# def calc_bbx_of_size(coord: tuple[float, float], size) -> (tuple[float, float], 'CRS'):
#     ''' 
#     Calculates the four corners of a bounding box
#     [bottom left, top right] as well as the UTM EPSG using Pyproj

#     Note: The input for this function is (x, y), not (lat, long)
#     '''
#     expansion = 10
    
#     inproj = Proj('epsg:4326')
#     outproj_code = calculate_epsg(coord)
#     outproj = Proj('epsg:' + str(outproj_code))

#     coord_utm = transform(inproj, outproj, coord[1], coord[0])
#     coord_utm_bottom_left = (coord_utm[0] - size // 2, coord_utm[1] - size // 2)

#     coord_utm_top_right = (coord_utm[0] + size // 2, expansion, coord_utm[1] + size // 2)
#     coord_bottom_left = transform(outproj, inproj, coord_utm_bottom_left[1], coord_utm_bottom_left[0])
#     coord_top_right = transform(outproj, inproj, coord_utm_top_right[1], coord_utm_top_right[0])
    
#     return (coord_bottom_left, coord_top_right)


# def calculate_epsg(points: Tuple[float, float]) -> int:
#     """ 
#     Calculates the UTM EPSG of an input WGS 84 lon, lat

#         Parameters:
#          points (tuple): input longitiude, latitude tuple

#         Returns:
#          epsg_code (int): integer form of associated UTM EPSG
#     """
#     lon, lat = points[0], points[1]
#     utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    
#     if len(utm_band) == 1:
#         utm_band = '0' + utm_band
    
#     if lat >= 0:
#         epsg_code = '326' + utm_band
    
#     else:
#         epsg_code = '327' + utm_band
    
#     return int(epsg_code)

def save_training_preds(arr: np.ndarray, tile_idx: tuple, country: str, suffix = "preds") -> str:
    '''
    Write predictions to a geotiff, using the same bounding box 
    to determine north, south, east, west corners of the tile
    '''
    
     # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
    out_folder = f'tmp/{country}/preds/'
    file = out_folder + f"{str(x)}X{str(y)}Y_{suffix}.tif"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    bbx = make_bbox(country, tile_idx)
    west, east = bbx[0], bbx[2]
    north, south = bbx[3], bbx[1]
    
    arr = arr.astype(np.uint8)

    # create the file based on the size of the array
    transform = rs.transform.from_bounds(west = west, south = south,
                                         east = east, north = north,
                                         width = arr.shape[1],
                                         height = arr.shape[0])

    print("Writing", file)
    new_dataset = rs.open(file, 'w', 
                            driver = 'GTiff',
                            height = arr.shape[0], width = arr.shape[1], count = 1,
                            dtype = "uint8",
                            compress = 'lzw',
                            crs = '+proj=longlat +datum=WGS84 +no_defs',
                            transform=transform)
    new_dataset.write(arr, 1)
    new_dataset.close()
    
    return None


## in progres
def make_gpro_friendly(raster_filepath):
    '''
    transforms a raster into format that can be imported into
    Google Earth Pro
    '''
    
    ds = gdal.Open(raster_filepath, 1)
    band = ds.GetRasterBand(1)
    colors = gdal.ColorTable()

    # set color for each value
    colors.SetColorEntry(0, (240, 247, 240)) 
    colors.SetColorEntry(1, (240, 247, 240))
    
    # set color table and color interpretation
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    # close and save file
    del band, ds
    
    return None