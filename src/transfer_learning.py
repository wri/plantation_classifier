import pandas as pd
import numpy as np
import hickle as hkl
import pickle
import seaborn as sns
import copy
import os
import boto3
import botocore
import rasterio as rs
import yaml
from osgeo import gdal
from scipy.ndimage import median_filter
from skimage.transform import resize
from glob import glob
import functools
from time import time, strftime
from datetime import datetime, timezone
from scipy import ndimage
from skimage.util import img_as_ubyte
import gc
import copy
import subprocess

## import other scripts
import sys
sys.path.append('src/')

import mosaic
import validate_io as validate
import slow_glcm as slow_txt
import fast_glcm as fast_txt

with open("config.yaml", 'r') as stream:
    document = (yaml.safe_load(stream))
    aak = document['aws']['aws_access_key_id']
    ask = document['aws']['aws_secret_access_key']


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

    dest_file = f'data/{location[1]}.csv'
    s3_file = f'2020/databases/{location[1]}.csv'

    # check if csv exists locally
    # confirm subdirectory exists otherwise download can fail
    if os.path.exists(dest_file):
        print(f'Csv file for {location[1]} exists locally.')
    
    if not os.path.exists('data/'):
        os.makedirs('data/')

    # if csv doesnt exist locally, check if available on s3
    if not os.path.exists(dest_file):
        s3 = boto3.resource('s3',
                            aws_access_key_id=aws_access_key, 
                            aws_secret_access_key=aws_secret_key)

        bucket = s3.Bucket('tof-output')
        
        # turn the bucket + file into a object summary list
        objs = list(bucket.objects.filter(Prefix=s3_file))
        print(s3_file, dest_file)

        if len(objs) > 0:
            print(f"The s3 resource s3://{bucket.name}/{s3_file} exists.")
            bucket.download_file(s3_file, dest_file)
            
    database = pd.read_csv(dest_file)

    # create a list of tiles 
    tiles = database[['X_tile', 'Y_tile']].to_records(index=False)

    return tiles

def download_ard(tile_idx: tuple, country: str, aws_access_key: str, aws_secret_key: str):
    ''' 
    If ARD folder is not present locally,
    Download contents from s3 folder into local folder
    for specified tile
    '''

    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]

    s3_path = f'2020/raw/{str(x)}/{str(y)}/'
    local_path = f'tmp/{country}/{str(x)}/{str(y)}/'

    # check if ARD folder has been downloaded
    # if not then download ARD and feats?
    # this might need to be updated to a single file endpoint
    ard_check = os.path.exists(local_path + 'ard/')

    if not ard_check:
        print(f"Downloading ARD & feats for {(x, y)}")

        s3 = boto3.resource('s3',
                            aws_access_key_id=aws_access_key, 
                            aws_secret_access_key=aws_secret_key)
        bucket = s3.Bucket('tof-output')

        # this needs to download whatever is in the ard folder + feats folder
        for obj in bucket.objects.filter(Prefix=s3_path):

            target = os.path.join(local_path, os.path.relpath(obj.key, s3_path))
            print(f'target download path: {target}')

            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == '/':
                continue
            try:
                bucket.download_file(obj.key, target)

            # if the tiles do not exist on s3, catch the error and return False
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False
    
    return True

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
    '''
    
    # prepare inputs
    x = tile_idx[0]
    y = tile_idx[1]
    folder = f'tmp/{country}/{str(x)}/{str(y)}/'
    tile_str = f'{str(x)}X{str(y)}Y'
    ard = hkl.load(f'{folder}ard/{tile_str}_ard.hkl')
    s2 = ard[..., 0:9]
    feats_file = f'{folder}raw/feats/{tile_str}_feats.hkl'
    feats_raw = hkl.load(feats_file).astype(np.float32)

    # prepare outputs
    # output shape will match s2 array ttc feats and 5 txt feats
    n_feats = len(feature_select) + 8 # for catv19 we have 8 txts
    output = np.zeros((s2.shape[0], s2.shape[1], n_feats), dtype=np.float32)

    # adjust TML predictions feats[0] to match training data (0-1)
    # adjust shape by rolling axis 2x (65, 614, 618) ->  (618, 614, 65) 
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

    # create no data and no tree flag (boolean mask)
    # where TML probability is 255 or 0, pass along to preds
    # note that the feats shape is (x, x, 65)
    no_data_flag = ttc[...,0] == 255.
    no_tree_flag = ttc[...,0] == 0.

    # apply feature selection to ttc feats 
    if len(feature_select) > 0:
        ttc = np.squeeze(ttc[:, :, [feature_select]])

    # import txt features if available, otherwise calc them
    if os.path.exists(f'{folder}raw/feats/{tile_str}_txtv19_ard.npy'):
        print('Importing texture features.')
        txt = np.load(f'{folder}raw/feats/{tile_str}_txtv19_ard.npy')

    else:
        s2 = img_as_ubyte(s2)
        assert s2.dtype == np.uint8, print(s2.dtype)
        
        txt = np.zeros((s2.shape[0], s2.shape[1], 8), dtype=np.float32)
        green = s2[..., 1]
        red = s2[..., 2]
        nir = s2[..., 3]
        print('Calculating select GLCM textures for green band...')
        txt[..., 0:4] = slow_txt.deply_extract_texture(green, ['dissimilarity', 'correlation', 'homogeneity', 'contrast'])
        print('Calculating select GLCM textures for red band...')
        txt[..., 4:5] = slow_txt.deply_extract_texture(red, ['contrast'])
        print('Calculating select GLCM textures for nir band...')
        txt[..., 5:] = slow_txt.deply_extract_texture(nir, ['dissimilarity', 'correlation', 'contrast'])

        # save glcm texture properties in case
        np.save(f'{folder}raw/feats/{tile_str}_txtv19.npy', txt)

    output[..., :ttc.shape[-1]] = ttc
    output[..., ttc.shape[-1]:] = txt

    del feats_raw, feats_rolled, high_feats, low_feats, ttc

    return output, no_data_flag, no_tree_flag

# This will need updating based on how ARD is stored
def make_sample(tile_idx: tuple, country: str, feats: np.array):
    
    ''' 
    Takes processed data, defines dimensions for the sample, then 
    combines dem, s1, s2 and features into a single array with shape (x, x, len(features))
    '''
    # prepare inputs
    x = tile_idx[0]
    y = tile_idx[1]

    ard = hkl.load(f'tmp/{country}/{str(x)}/{str(y)}/ard/{str(x)}X{str(y)}Y_ard.hkl')

    # define number of features in the sample
    n_feats = 1 + ard.shape[-1] + feats.shape[-1] 

    # Create the empty array using shape of inputs
    sample = np.zeros((ard.shape[0], ard.shape[1], n_feats), dtype=np.float32)
    
    # populate empty array with each feature
    sample[..., 0:13] = ard
    sample[..., 13:] = feats

    # save dims for future use
    arr_dims = (sample.shape[0], sample.shape[1])

    return sample, arr_dims

def reshape(arr: np.array, verbose: bool = False):

    ''' 
    Do not apply scaling and only reshape the unseen data.
    '''

    arr_reshaped = np.reshape(arr, (np.prod(arr.shape[:-1]), arr.shape[-1]))

    if verbose:
        print(arr_reshaped.shape)

    return arr_reshaped

def predict_classification(arr: np.array, pretrained_model: str, sample_dims: tuple):

    '''
    Import pretrained model and run predictions on arr.
    If using a regression model, multiply results by 100
    to get probability 0-100. Reshape array to permit writing to tif.
    '''
    
    preds = pretrained_model.predict(arr)
    
    # TODO: update pipeline for regression
    # if 'rfr' in model:
    #     preds = preds * 100

    return preds.reshape(sample_dims[0], sample_dims[1])

def post_process_tile(arr: np.array, feature_select: list, no_data_flag: np.array, no_tree_flag: np.array, thresh=10):

    '''
    Applies the no data and no tree flag *if* TTC predictions are used
    in feature selection. 
    Performs a connected component analysis to remove positive predictions 
    where the connected pixel count is < thresh. Establishing a minimum 
    plantation size (0.1 ha?)will remove the "noisy" pixels
    '''

    # flag - this wouldnt apply if all feats used
    # getting float 32 prediction from neural network and it's rarely going to be 0.
    # will be a continuous value
    # should be less than 0.1
    if 0 in feature_select:
        arr[no_data_flag] = 255.
        arr[no_tree_flag] = 0.

    # returns a labeled array, where each unique feature has a unique label
    # returns how many objects were found
    # nlabels is number of unique patches
    Zlabeled, Nlabels = ndimage.label(arr)
    
    # get pixel count for each label
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
    
    # if the count of pixels doesn't meet the threshold, make label 0
    for label,size in enumerate(label_size):
        if size < thresh:
            arr[Zlabeled == label] = 0
    
    # TODO: for monoculture, must be greater than x connected pixels
  
    del Zlabeled, Nlabels, label_size

    return arr

def write_tif(arr: np.ndarray, bbx: list, tile_idx: tuple, country: str, suffix = "preds") -> str:
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
    
    del arr, new_dataset

    return None

def remove_folder(tile_idx: tuple, local_dir: str):
    '''
    Deletes temporary raw data files in path_to_tile/raw/*
    after predictions are written to file
    '''
    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
  
    path_to_tile = f'{local_dir}/{str(x)}/{str(y)}/'

    # remove every folder/file in raw/
    for folder in glob(path_to_tile + "raw/*/"):
        for file in os.listdir(folder):
            _file = folder + file
            os.remove(_file)
        
    return None

def execute_per_tile(tile_idx: tuple, location: list, model, verbose: bool, feature_select: list):
    
    print(f'Processing tile: {tile_idx}')
    successful = download_ard(tile_idx, location[0], aak, ask)

    if successful:
        feats, no_data_flag, no_tree_flag = process_feats_slow(tile_idx, location[0], feature_select)
        sample, sample_dims = make_sample(tile_idx, location[0], feats)
        sample_ss = reshape(sample, verbose)
        
        validate.model_inputs(sample_ss)
        preds = predict_classification(sample_ss, model, sample_dims)
        preds_final = post_process_tile(preds, feature_select, no_data_flag, no_tree_flag)

        #validate.classification_scores(preds)
        write_tif(preds_final, tile_idx, location[0], 'preds_ARD')
        #remove_folder(tile_idx, local_dir)

        # clean up memory
        del feats, no_data_flag, no_tree_flag, sample, sample_ss, preds, preds_final
    
    else:
        print(f'Raw data for {tile_idx} does not exist on s3.')

    return None


if __name__ == '__main__':
   
    import argparse
    parser = argparse.ArgumentParser()
    print("Argument List:", str(sys.argv))

    parser.add_argument('--loc', dest='location', nargs='+', type=str)
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--verbose', dest='verbose', default=False, type=bool) 
    parser.add_argument('--incl_feats', dest='incl_feats', default=True, type=bool) 
    parser.add_argument('--fs', dest='feature_select', nargs='*', type=int) 


    args = parser.parse_args()
    
    # specify tiles HERE
    tiles_to_process = download_tile_ids(args.location, aak, ask)
    tile_count = len(tiles_to_process)
    counter = 0

    # load specified model
    with open(f'models/{args.model}.pkl', 'rb') as file:  
        loaded_model = pickle.load(file)

    print('............................................')
    print(f'Processing {tile_count} tiles for {args.location[1], args.location[0]}.')
    print('............................................')

    for tile_idx in tiles_to_process:
        counter += 1
        execute_per_tile(tile_idx, location=args.location, model=loaded_model, verbose=args.verbose, incl_feats=args.incl_feats, feature_select=args.feature_select)

        if counter % 5 == 0:
            print(f'{counter}/{tile_count} tiles processed...')
    
    # for now mosaic and upload to s3 bucket
    #mosaic.mosaic_tif(args.location, args.model, compile_from='csv')
    #mosaic.upload_mosaic(args.loc, args.model, aak, ask)
    