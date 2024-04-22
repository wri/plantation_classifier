#!/usr/bin/env python

import pandas as pd
import numpy as np
import hickle as hkl
import seaborn as sns
import copy
import os
import boto3
import botocore
import rasterio as rs
from rasterio.plot import reshape_as_raster
import yaml
from osgeo import gdal
from skimage.transform import resize
from glob import glob
import functools
from time import time, strftime
from datetime import datetime
import joblib
import copy
import utils.validate_io as validate
import utils.postprocess as postprocess
from utils import mosaic
import shutil

def timer(func):
    '''
    Prints the runtime of the decorated function.
    '''
    
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = datetime.now() 
        value = func(*args, **kwargs)
        end = datetime.now() 
        run_time = end - start
        print(f'Completed {func.__name__!r} in {run_time}.')
        return value
    return wrapper_timer

def download_tile_ids(location: list, aws_access_key: str, aws_secret_key: str):
    '''
    Checks to see if a country csv file exists locally,
    if not downloads the file from s3 and creates
    a list of tiles for processing.
    '''
    local = params['deploy']['data_dir']
    s3_file = f'2020/databases/{location[1]}.csv'
    dest_file = f"{local}database/{location[1]}.csv"

    # check if csv exists locally
    # confirm subdirectory exists otherwise download can fail
    if os.path.exists(dest_file):
        print(f'Csv file for {location[1]} exists locally.')
    
    if not os.path.exists(local):
        os.makedirs(local)

    # if csv doesnt exist locally, check if available on s3
    if not os.path.exists(dest_file):
        s3 = boto3.resource('s3',
                            aws_access_key_id=aws_access_key,
                            aws_secret_access_key=aws_secret_key)

        bucket = s3.Bucket(params['deploy']['bucket'])
        
        # turn the bucket + file into a object summary list
        objs = list(bucket.objects.filter(Prefix=s3_file))
        print(s3_file, dest_file)

        if len(objs) > 0:
            print(f"The s3 resource s3://{bucket.name}/{s3_file} exists.")
            bucket.download_file(s3_file, dest_file)
            
    database = pd.read_csv(dest_file)

    # create a list of tiles 
    tiles = database[['X_tile', 'Y_tile']].to_records(index=False)

    return database, tiles

def download_ard(tile_idx: tuple, country: str, aws_access_key: str, aws_secret_key: str, overwrite: bool):
    ''' 
    If ARD folder or the feats folder are not present locally,
    Download contents from s3 into local for specified tile
    If overwrite is set to True, the ARD and feats dir (txt and ttc feats) will be 
    redownloaded regardless of whether the file is present locally.
    '''
    x = tile_idx[0]
    y = tile_idx[1]

    s3_path_ard = f'2020/ard/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_ard.hkl'
    s3_path_feats = f'2020/raw/{str(x)}/{str(y)}/raw/feats/'

    # check if present locally
    # for ARD checks for single file, for feats checks for folder
    local_path = f'tmp/{country}/{str(x)}/{str(y)}/'
    local_ard = os.path.exists(f'{local_path}ard/{str(x)}X{str(y)}Y_ard.hkl')
    local_feats = os.path.exists(local_path + 'raw/feats/')

    s3 = boto3.resource('s3',
                    aws_access_key_id=aws_access_key, 
                    aws_secret_access_key=aws_secret_key)
    bucket = s3.Bucket(params['deploy']['bucket'])

    if local_ard == False or overwrite == True:
        print(f"Downloading ARD for {(x, y)}")
        if not os.path.exists(local_path + 'ard/'):
            os.makedirs(local_path + 'ard/')
    
        try:
            bucket.download_file(s3_path_ard, f'{local_path}ard/{str(x)}X{str(y)}Y_ard.hkl')

        except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False
        
    if local_feats == False or overwrite == True:
        print(f"Downloading feats for {(x, y)}")

        for obj in bucket.objects.filter(Prefix=s3_path_feats):

            feats_target = os.path.join(local_path + 'raw/feats/', os.path.relpath(obj.key, s3_path_feats))

            if not os.path.exists(os.path.dirname(feats_target)):
                os.makedirs(os.path.dirname(feats_target))
            if obj.key[-1] == '/':
                continue
            try:
                bucket.download_file(obj.key, feats_target)

            # if the tiles do not exist on s3, catch the error and return False
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False
    return True

def make_bbox(bbx_df, tile_idx: tuple, expansion: int = 10) -> list:
    """
    Makes a (min_x, min_y, max_x, max_y) bounding box that
    is 2 * expansion 300 x 300 meter ESA LULC pixels. 

       Parameters:
            initial_bbx (list): [min_x, min_y, max_x, max_y]
            expansion (int): 1/2 number of 300m pixels to expand

       Returns:
            bbx (list): expanded [min_x, min_y, max_x, max_y]
    """
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
    
    return bbx

def process_feats_slow(tile_idx: tuple, country: str, feature_select:list) -> np.ndarray:
    '''
    Applies preprocessing steps to the TTC feats extracted from the CNN:
        - starting shape (65, x, x)
        - scale tree prediction (feats[0]) between 0-1 to match the training
          pipeline 
        - roll the axis to adjust shape
        - swap high and low level feats to match training pipeline
        - filter to selected feats if feature_select param > 0
        - creates no data and no tree flags for masking predictions
    Combines TTC feats with texture properties and returns array filtered
    to selected feats

    '''
    # prepare inputs
    x = tile_idx[0]
    y = tile_idx[1]
    folder = f'tmp/{country}/{str(x)}/{str(y)}/raw/feats/'
    tile_str = f'{str(x)}X{str(y)}Y'
    feats_raw = hkl.load(f'{folder}{tile_str}_feats.hkl').astype(np.float32)
    txt = np.load(f'{folder}{tile_str}_txt.npy')

    # prepare outputs 
    n_feats = 65 + 16
    output = np.zeros((txt.shape[0], txt.shape[1], n_feats), dtype=np.float32)

    # adjust TML predictions feats[0] to match training data (0-1)
    # adjust shape by rolling axis 2x (65, 614, 618) ->  (614, 618, 65), (618, 614, 65) 
    # feats used for deply are multiplyed by 1000 before saving
    feats_raw[0, ...] = feats_raw[0, ...] / 100 
    feats_raw[1:, ...] = feats_raw[1:, ...] / 1000  
    feats_rolled = np.rollaxis(feats_raw, 0, 3)
    feats_rolled = np.rollaxis(feats_rolled, 0, 2)
    
    # now switch the feats
    ttc = copy.deepcopy(feats_rolled) 

    high_feats = [np.arange(1,33)]
    low_feats = [np.arange(33,65)]

    ttc[:, :, [low_feats]] = feats_rolled[:, :, [high_feats]]
    ttc[:, :, [high_feats]] = feats_rolled[:, :, [low_feats]]

    # combine ttc feats and txt into a single array
    output[..., :ttc.shape[-1]] = ttc
    output[..., ttc.shape[-1]:] = txt
    
    # apply feature selection - filtered to non ARD indices
    feature_select = [i - 13 for i in feature_select if i >= 13]
    if len(feature_select) > 0:
        output = np.squeeze(output[:, :, [feature_select]])

    del feats_raw, feats_rolled, high_feats, low_feats

    return output, ttc

def make_sample(tile_idx: tuple, country: str, feats: np.array):
    
    ''' 
    Takes processed data, defines dimensions for the sample, then 
    combines ard and feats
    ard (618, 614, 13)
    feats (618, 614, 40)
    sample (618, 614, 53)
    '''
    # prepare inputs
    x = tile_idx[0]
    y = tile_idx[1]
    ard = hkl.load(f'tmp/{country}/{str(x)}/{str(y)}/ard/{str(x)}X{str(y)}Y_ard.hkl')
    ard = np.clip(ard, 0, 1)

    # define number of features and create sample array
    n_feats = ard.shape[-1] + feats.shape[-1] 
    sample = np.zeros((ard.shape[0], ard.shape[1], n_feats), dtype=np.float32)
    
    # populate empty array with each feature
    # order: s2, dem, s1, ttc, txt
    sample[..., 0:10] = ard[..., 0:10]
    sample[..., 10:11] = ard[..., 10:11]
    sample[..., 11:13] = ard[..., 11:13]
    sample[..., 13:] = feats

    # save dims for future use
    arr_dims = (sample.shape[0], sample.shape[1])

    return sample, arr_dims

def reshape(arr: np.array, verbose: bool):

    ''' 
    Do not apply scaling and only reshape the unseen data.
    '''

    arr_reshaped = np.reshape(arr, (np.prod(arr.shape[:-1]), arr.shape[-1]))

    if verbose:
        print(arr_reshaped.shape)

    return arr_reshaped

def reshape_and_scale(v_train_data: str, unseen: np.array, verbose: bool = False):

    ''' 
    Manually standardizes the unseen array on a 1%, 99% scaler 
    instead of applying a mean, std scaler (StandardScalar). Imports array of mins/maxs from
    appropriate training dataset for scaling of unseen data. Then reshapes the sample from 
    (x, x, 13 or 78) to (x, 13 or 78).
    '''

    # import mins and maxs from appropriate training dataset 
    mins = np.load(f'data/mins_{v_train_data}.npy')
    maxs = np.load(f'data/maxs_{v_train_data}.npy')
    start_min, start_max = unseen.min(), unseen.max()

    # iterate through the bands and standardize the data based 
    # on a 1 and 99% scaler instead of a mean and std scaler (standard scalar)
    for band in range(0, unseen.shape[-1]):

        min = mins[band]
        max = maxs[band]
        if verbose:
            print(f'Band {band}: {min} - {max}')

        if max > min:
            
            # clip values outside of min - max interval for unseen band
            unseen[..., band] = np.clip(unseen[..., band], min, max)

            # now calculate standardized data for unseen
            midrange = (max + min) / 2
            rng = max - min
            standardized = (unseen[..., band] - midrange) / (rng / 2)

            # update each band in unseen to the standardized data
            unseen[..., band] = standardized
            end_min, end_max = unseen.min(), unseen.max()
            
        else:
            print('Warning: mins > maxs')
            pass

    # now reshape
    unseen_reshaped = np.reshape(unseen, (np.prod(unseen.shape[:-1]), unseen.shape[-1]))

    return unseen_reshaped

def predict_classification(arr: np.array, pretrained_model: str, sample_dims: tuple):

    '''
    Import pretrained model and run predictions on arr.
    Reshape array to permit writing to tif.
    '''

    preds = pretrained_model.predict(arr)
    preds = preds.reshape(sample_dims[0], sample_dims[1])

    return preds

def predict_regression(arr: np.array, pretrained_model: str, sample_dims: tuple):
    '''
    Import pretrained model and run predictions on arr.
    If using a regression model, multiply results by 100
    to get probability 0-100. Reshape array to permit writing to tif.

    model.predict() outputs a 2D array with shape (379452, 2). This can
    only be reshaped to (618, 614, 2). But there are 3 classes, so how
    to get the model to predict probability for each class?
    '''
        
    #preds = pretrained_model.predict(arr, prediction_type='Probability') 
    preds = pretrained_model.predict_proba(arr)
    preds = preds * 100
    print(f'original shape: {preds.shape}')
    preds = preds.reshape((sample_dims[0], sample_dims[1], 3))
    print(f'reshaped: {preds.shape}')

    return preds

def write_tif(arr: np.ndarray, bbx: list, tile_idx: tuple, country: str, model_type: str, suffix = "preds"):
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

    # uses bbx to figure out the corners
    west, east = bbx[0], bbx[2]
    north, south = bbx[3], bbx[1]
    
    arr = arr.astype(np.uint8)

    # create the file based on the size of the array (618, 614, 1)
    if model_type == 'classifier':
        transform = rs.transform.from_bounds(west = west, south = south,
                                            east = east, north = north,
                                            width = arr.shape[1],  #614
                                            height = arr.shape[0]) #618
        new_dataset = rs.open(file, 'w', 
                                driver = 'GTiff',
                                width = arr.shape[1], 
                                height = arr.shape[0], 
                                count = 1,
                                dtype = "uint8",
                                compress = 'lzw',
                                crs = '+proj=longlat +datum=WGS84 +no_defs',
                                transform=transform)
        new_dataset.write(arr, 1)
        new_dataset.close()
    
    # switch (618, 614, band) to (band, 618, 614)
    else:
        arr = reshape_as_raster(arr)
        print(f'reshaped #2: {arr.shape}') 
        transform = rs.transform.from_bounds(west = west, south = south,
                                            east = east, north = north,
                                            width = arr.shape[2], 
                                            height = arr.shape[1])
        
        new_dataset = rs.open(file,'w',
                            driver='GTiff',
                            width=arr.shape[2],
                            height=arr.shape[1],
                            count=arr.shape[0],
                            dtype = "uint8",
                            compress = 'lzw',
                            crs = '+proj=longlat +datum=WGS84 +no_defs',
                            transform=transform)
        new_dataset.write(arr) # adding count here throws error
        new_dataset.close()

    del arr, new_dataset

    return None

def remove_folder(tile_idx: tuple, location: str):
    '''
    Deletes temporary raw data files in path_to_tile/raw/*
    after predictions are written to file
    '''
    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
  
    print(f"Removing raw files for {tile_idx}")
    dir_to_delete = f'tmp/{location}/{str(x)}/{str(y)}/'
    shutil.rmtree(dir_to_delete)
        
    return None

def execute_per_tile(database, tile_idx: tuple, location: list, model, verbose: bool, feature_select: list, model_type: str, overwrite: bool):

    ''' 
    Execute all steps in the pipeline for the given tile.
    '''
    print(f'Processing tile: {tile_idx}')
    successful = download_ard(tile_idx, location[0], aak, ask, overwrite)

    if successful:
        x = tile_idx[0]
        y = tile_idx[1]
        ard = hkl.load(f'tmp/{location[0]}/{str(x)}/{str(y)}/ard/{str(x)}X{str(y)}Y_ard.hkl')
        validate.input_ard(tile_idx, location[0])
        bbx = make_bbox(database, tile_idx)
        validate.output_dtype_and_dimensions(ard[..., 11:13], ard[..., 0:10], ard[..., 10])
        validate.feats_range(tile_idx, location[0])
        feats, ttc = process_feats_slow(tile_idx, location[0], feature_select)
        sample, sample_dims = make_sample(tile_idx, location[0], feats)
        sample_ss = reshape(sample, verbose)
        #sample_ss = reshape_and_scale('v20', sample, verbose)
        
        validate.model_inputs(sample_ss)
        if model_type == 'classifier':
            preds = predict_classification(sample_ss, model, sample_dims)
            preds_final = postprocess.clean_tile(preds, feature_select, ttc)
            validate.model_outputs(preds, model_type)
        else:
            preds_final = predict_regression(sample_ss, model, sample_dims)

        write_tif(preds_final, bbx, tile_idx, location[0], model_type, 'preds')
        if params['deploy']['cleanup']:
            remove_folder(tile_idx, location[0])

        del ard, feats, sample, sample_ss, preds_final
    
    else:
        print(f'Raw data for {tile_idx} could not be downloaded or does not exist on s3.')
        return None

    return None


if __name__ == '__main__':
   
    import argparse
    import json
    parser = argparse.ArgumentParser()    
    parser.add_argument('--params', dest='params', required=True) 
    parser.add_argument('--loc', dest='location', nargs='+', type=str)
    parser.add_argument('--slicer', nargs='+', type=int)
    args = parser.parse_args()

    print(f'Initializing...............................\n')

    with open(args.params) as param_file:
        params = yaml.safe_load(param_file)
    with open(params['base']['config']) as conf_file:
        config = yaml.safe_load(conf_file)
        aak = config['aws']['aws_access_key_id']
        ask = config['aws']['aws_secret_access_key']
    
    location = args.location
    model_type = params['deploy']['model_type']
    print(f'Model type is {model_type}.')
    with open(params['deploy']['model_path'], "rb") as fp:
        loaded_model = joblib.load(fp)
    with open(params["select"]["selected_features_path"], "r") as fp:
        selected_features = json.load(fp)

    # specify tiles to process
    database, tile_ids = download_tile_ids(location, aak, ask)
    indices = tuple(args.slicer)
    if len(indices) == 1 and indices[0] == -1:
        tiles_to_process = tile_ids
    else: 
        tiles_to_process = tile_ids[slice(*indices)] # unpack tuple with *
    tile_count = len(tiles_to_process)

    print('............................................')
    print(f'Processing {tile_count} tiles for {location[1]}, {location[0]}')
    print('............................................')
    
    counter = 0
    for tile_idx in tiles_to_process:
        counter += 1
        execute_per_tile(database,
                         tile_idx, 
                         location, 
                         loaded_model, 
                         params['deploy']['verbose'], 
                         selected_features, 
                         model_type,
                         params['deploy']['overwrite'])

        if counter % 2 == 0:
            print(f'{counter}/{tile_count} tiles processed...')
    
    mosaic.mosaic_tif(location, params['deploy']['version'], tiles_to_process)
    file_to_upload = mosaic.clip_it(aak, ask, 
                                    location, 
                                    params['deploy']['version'], 
                                    params['deploy']['data_dir'],
                                    params['deploy']['bucket'])
    mosaic.upload_mosaic(aak, ask, file_to_upload)
  
    
