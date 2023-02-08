#! /usr/bin/env python3

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
from datetime import datetime

import sys
sys.path.append('src/')
import interpolation
import cloud_removal
import mosaic
import validate_io as validate


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


## Step 1: Download raw data from s3
def download_tile_ids(country: str, aws_access_key: str, aws_secret_key: str):
    '''
    Checks to see if a country csv file exists locally,
    if not downloads the file from s3 and creates
    a list of tiles for processing.
    '''
    dest_file = f'data/{country}.csv'
    s3_file = f'2020/databases/{country}.csv'

    # check if csv exists locally
    # confirm subdirectory exists otherwise download can fail
    if os.path.exists(dest_file):
        print('Csv file exists locally.')
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


def download_folder(s3_folder: str, local_dir: str, aws_access_key: str, aws_secret_key: str):
    '''
    Download the contents of the tof-output + s3-folder 
    into a local folder.
    '''
    
    s3 = boto3.resource('s3',
                        aws_access_key_id=aws_access_key, 
                        aws_secret_access_key=aws_secret_key)

    bucket = s3.Bucket('tof-output')

    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
        
    return None


def download_raw_tile(tile_idx: tuple, local_dir: str, access_key: str, secret_key: str) -> None:
    '''
    If data is not present locally, downloads raw data (clouds, DEM and 
    image dates, s1 and s2 (10 and 20m bands)) for the specified tile from s3. 
    Not free to run downloads to local - use sparingly.
    '''

    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
    
    # state local path and s3
    path_to_tile = f'{local_dir}/{str(x)}/{str(y)}/'
    s3_path_to_tile = f'2020/raw/{str(x)}/{str(y)}/'
    
    # check if feats folder exists locally
    folder_check = os.path.exists(path_to_tile + f"raw/feats/{str(x)}X{str(y)}Y_feats.hkl")

    if folder_check:
        print('Raw data exists locally.')
        return True
    
    # if feats folder doesn't exist locally, download raw tile
    # and return True
    if not folder_check:
        print(f"Downloading data for {(x, y)}")
        try: 
            download_folder(s3_folder = s3_path_to_tile,
                            local_dir = path_to_tile,
                            aws_access_key = access_key,
                            aws_secret_key = secret_key)

            return True

        # if the tiles do not exist on s3, catch the error and return False
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False


## Step 2: Create a cloud free composte

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
    data = pd.read_csv(f"data/{country}.csv")

    # this will remove quotes around x and y tile indexes (not needed for all countries)
    # data['X_tile'] = data['X_tile'].str.extract('(\d+)', expand=False)
    # data['X_tile'] = pd.to_numeric(data['X_tile'])
    # data['Y_tile'] = data['Y_tile'].str.extract('(\d+)', expand=False)
    # data['Y_tile'] = pd.to_numeric(data['Y_tile'])

    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]

    # make a copy of the database 
    bbx_df = data.copy()
    
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

def convert_to_db(x: np.ndarray, min_db: int) -> np.ndarray:
    """ 
    Converts Sentinel 1 unitless backscatter coefficient
    to db with a min_db lower threshold
    
    Parameters:
        x (np.ndarray): unitless backscatter (T, X, Y, B) array
        min_db (int): integer from -50 to 0

    Returns:
        x (np.ndarray): db backscatter (T, X, Y, B) array
    """
    
    x = 10 * np.log10(x + 1/65535)
    x[x < -min_db] = -min_db
    x = (x + min_db) / min_db
    return np.clip(x, 0, 1)

def to_float32(array: np.array) -> np.array:
    """
    Ensures input array is not already a float and does not range from 0-1, 
    then converts int16 to float32.

    Data is stored as uint16 (0-65535), so must divide by 65535 to get on 0-1 scale.
    Using assertions to ensure the division only happens when it needs to, 
    i.e. why we use this function instead of just calling np.float32(array) 
    """

    # print(f'The original max value is {np.max(array)}')
    if not isinstance(array.flat[0], np.floating):
        assert np.max(array) > 1
        array = np.float32(array) / 65535.
    assert np.max(array) <= 1
    assert array.dtype == np.float32
    
    return array

def adjust_shape(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Ensures that the shape of arr is width x height.
    Used to align 10, 20, 40, 160, 640 meter resolution Sentinel data
    Applied to s1, s2 and dem data in process_tile()
    """
    # print(f"Input array shape: {arr.shape}")
    arr = arr[:, :, :, np.newaxis] if len(arr.shape) == 3 else arr
    arr = arr[np.newaxis, :, :, np.newaxis] if len(arr.shape) == 2 else arr
    
    if arr.shape[1] < width:
        pad_amt = (width - arr.shape[1]) // 2
        if pad_amt == 0:
            arr = np.pad(arr, ((0, 0), (1, pad_amt), (0,0), (0, 0)), 'edge')
        else:
            arr = np.pad(arr, ((0, 0), (pad_amt, pad_amt), (0,0), (0, 0)), 'edge')

    if arr.shape[2] < height:
        pad_amt = (height - arr.shape[2]) // 2
        if pad_amt == 0:
            arr = np.pad(arr, ((0, 0), (0,0), (1, 0), (0, 0)), 'edge')
        else:
            arr = np.pad(arr, ((0, 0), (0,0), (pad_amt, pad_amt), (0, 0)), 'edge')

    if arr.shape[1] > width:
        pad_amt =  (arr.shape[1] - width) // 2
        pad_amt_even = (arr.shape[1] - width) % 2 == 0
        if pad_amt == 0:
            arr = arr[:, 1:, ...]
        elif pad_amt_even:
            pad_amt = int(pad_amt)
            arr = arr[:, pad_amt:-pad_amt, ...]
        else:
            pad_left = int(np.floor(pad_amt / 2))
            pad_right = int(np.ceil(pad_amt / 2))
            arr = arr[:, pad_left:-pad_right, ...]

    if arr.shape[2] > height:
        pad_amt = (arr.shape[2] - height) // 2
        pad_amt_even = (arr.shape[2] - height) % 2 == 0
        if pad_amt == 0:
            arr = arr[:, :, 1:, :]
        elif pad_amt_even:
            pad_amt = int(pad_amt)
            arr = arr[:, :, pad_amt:-pad_amt, ...]
        else:
            pad_left = int(np.floor(pad_amt / 2))
            pad_right = int(np.ceil(pad_amt / 2))
            arr = arr[:, :, pad_left:-pad_right, ...]

    return arr.squeeze()

def process_tml_feats(tile_idx: tuple, local_path: str, feats: bool, feature_select:list) -> np.ndarray:
    '''
    Transforms the feats extracted from the TML model (in temp/raw/tile_feats..) to 
    processed data structure
        - scale tree prediction (feats[0]) between 0-1 to match the training
          pipeline 
        - roll the axis to adjust shape
        - swap high and low level feats to match training pipeline
        - filter to selected feats if feature_select param > 0
        - creates no data and no tree flags for masking predictions
    '''

    x = str(tile_idx[0])
    y = str(tile_idx[1])
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y

    folder = f"{local_path}/{str(x)}/{str(y)}/"
    tile_str = f'{str(x)}X{str(y)}Y'

    # load and prep features here
    if feats:
        feats_file = f'{folder}raw/feats/{tile_str}_feats.hkl'
        feats_raw = hkl.load(feats_file).astype(np.float32)
    
        # adjust TML predictions feats[0] to match training data (0-1)
        # adjust shape by rolling axis (65, 614, 618) ->  (618, 614, 65) 
        feats_raw[0] = feats_raw[0] / 100 
        feats_raw[1:] = feats_raw[1:] / 1000  # feats are multiplyed by 1000 before saving
        feats_rolled = np.rollaxis(feats_raw, 0, 3)
        feats_rolled = np.rollaxis(feats_rolled, 0, 2)

        # now switch the feats
        feats_ = feats_rolled.copy()

        high_feats = [np.arange(1,33)]
        low_feats = [np.arange(33,65)]

        feats_[:, :, [low_feats]] = feats_rolled[:, :, [high_feats]]
        feats_[:, :, [high_feats]] = feats_rolled[:, :, [low_feats]]

        # create no data and no tree flag (boolean mask)
        # where TML probability is 255 or 0, pass along to preds
        # note that the feats shape is (x, x, 65)
        no_data_flag = feats_[...,0] == 255.
        no_tree_flag = feats_[...,0] == 0.

        # if only using select feats, np.take will take elements from an array along an axis
        if len(feature_select) > 0:
            feats_ = np.squeeze(feats_[:, :, [feature_select]])

    # remove this else statement once pipeline updated to feat only 
    else:
        feats_ = []

    return feats_, no_data_flag, no_tree_flag



def process_tile(tile_idx: tuple, local_path: str, bbx: list, verbose: bool = False, make_shadow: bool = True) -> np.ndarray:
    """
    Transforms raw data structure (in temp/raw/*) to processed data structure
        - align shapes of different data sources (clouds / shadows / s1 / s2 / dem)
        - superresolve 20m to 10m with bilinear upsampling for DSen2 input
        - remove (interpolate) clouds and shadows
    Parameters:
         x (int): x position of tile to be downloaded
         y (int): y position of tile to be downloaded
         data (pd.DataFrame): tile grid dataframe (note: not accessed)
         bbx: bounding box
        Returns:
         x (np.ndarray)
         image_dates (np.ndarray)
         interp (np.ndarray)
         s1 (np.ndarray)
         s2 (np.ndarray)
    """
    
    x = str(tile_idx[0])
    y = str(tile_idx[1])
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y
            
    folder = f"{local_path}/{str(x)}/{str(y)}/"
    tile_str = f'{str(x)}X{str(y)}Y'

    clouds_file = f'{folder}raw/clouds/clouds_{tile_str}.hkl'
    cloud_mask_file = f'{folder}raw/clouds/cloudmask_{tile_str}.hkl'
    shadows_file = f'{folder}raw/clouds/shadows_{tile_str}.hkl'
    s1_file = f'{folder}raw/s1/{tile_str}.hkl'
    s1_dates_file = f'{folder}raw/misc/s1_dates_{tile_str}.hkl'
    s2_10_file = f'{folder}raw/s2_10/{tile_str}.hkl'
    s2_20_file = f'{folder}raw/s2_20/{tile_str}.hkl'
    s2_dates_file = f'{folder}raw/misc/s2_dates_{tile_str}.hkl'
    clean_steps_file = f'{folder}raw/clouds/clean_steps_{tile_str}.hkl'
    dem_file = f'{folder}raw/misc/dem_{tile_str}.hkl'
    
    clouds = hkl.load(clouds_file)

    if os.path.exists(cloud_mask_file):
        # These are the S2Cloudless / Sen2Cor masks
        clm = hkl.load(cloud_mask_file).repeat(2, axis = 1).repeat(2, axis = 2)
    else:
        clm = None

    s1 = hkl.load(s1_file)
    s1 = np.float32(s1) / 65535
    s1[..., -1] = convert_to_db(s1[..., -1], 22)
    s1[..., -2] = convert_to_db(s1[..., -2], 22)
    s1 = s1.astype(np.float32)
    
    s2_10 = to_float32(hkl.load(s2_10_file))
    s2_20 = to_float32(hkl.load(s2_20_file))
    
    # slice off last index (no data mask) if present
    if s2_20.shape[3] == 7:
        s2_20 = s2_20[..., :6]
        #print(f's2_20 data mask removed.')

    dem = hkl.load(dem_file)
    dem = median_filter(dem, size = 5)
    image_dates = hkl.load(s2_dates_file)
    
    # Ensure arrays are the same dims
    width = s2_20.shape[1] * 2
    height = s2_20.shape[2] * 2
    s1 = adjust_shape(s1, width, height)
    s2_10 = adjust_shape(s2_10, width, height)
    dem = adjust_shape(dem, width, height)

    # Deal with cases w/ only 1 image
    if len(s2_10.shape) == 3:
        s2_10 = s2_10[np.newaxis]
    if len(s2_20.shape) == 3:
        s2_20 = s2_20[np.newaxis]

    if verbose:
        print(f'Clouds: {clouds.shape}, \n'
                f'S1: {s1.shape} \n'
                f'S2: {s2_10.shape}, {s2_20.shape} \n'
                f'DEM: {dem.shape}')

    # bilinearly upsample 20m bands to 10m for superresolution
    sentinel2 = np.zeros((s2_10.shape[0], width, height, 10), np.float32)
    sentinel2[..., :4] = s2_10

    # a foor loop is faster than trying to vectorize it here! 
    for band in range(4):
        for step in range(sentinel2.shape[0]):
            sentinel2[step, ..., band + 4] = resize(
                s2_20[step,..., band], (width, height), 1)

    for band in range(4, 6):
        # indices 4, 5 are 40m and may be a different shape
        # this code is ugly, but it forces the arrays to match up w/ the 10/20m ones
        for step in range(sentinel2.shape[0]):
            mid = s2_20[step,..., band]
            if (mid.shape[0] % 2 == 0) and (mid.shape[1] % 2) == 0:
                mid = mid.reshape(mid.shape[0] // 2, 2, mid.shape[1] // 2, 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, ..., band + 4] = resize(mid, (width, height), 1)
            if mid.shape[0] %2 != 0 and mid.shape[1] %2 != 0:
                mid_misaligned_x = mid[0, :]
                mid_misaligned_y = mid[:, 0]
                mid = mid[1:, 1:].reshape(
                    int(np.floor(mid.shape[0] / 2)), 2,
                    int(np.floor(mid.shape[1] / 2)), 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, 1:, 1:, band + 4] = resize(mid, (width - 1, height - 1), 1)
                sentinel2[step, 0, :, band + 4] = mid_misaligned_x.repeat(2)
                sentinel2[step, :, 0, band + 4] = mid_misaligned_y.repeat(2)
            elif mid.shape[0] % 2 != 0:
                mid_misaligned = mid[0, :]
                mid = mid[1:].reshape(int(np.floor(mid.shape[0] / 2)), 2, mid.shape[1] // 2, 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, 1:, :, band + 4] = resize(mid, (width - 1, height), 1)
                sentinel2[step, 0, :, band + 4] = mid_misaligned.repeat(2)
            elif mid.shape[1] % 2 != 0:
                mid_misaligned = mid[:, 0]
                mid = mid[:, 1:]
                mid = mid.reshape(mid.shape[0] // 2, 2, int(np.floor(mid.shape[1] / 2)), 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, :, 1:, band + 4] = resize(mid, (width, height - 1), 1)
                sentinel2[step, :, 0, band + 4] = mid_misaligned.repeat(2)

    # Identifies missing imagery (either in sentinel acquisition, or induced in preprocessing)
    # If more than 50% of data for a time step is missing, then remove them....
    missing_px = interpolation.id_missing_px(sentinel2, 2)
    if len(missing_px) > 0:
        #print(f"Removing {missing_px} dates due to {missing_px} missing data")
        clouds = np.delete(clouds, missing_px, axis = 0)
        image_dates = np.delete(image_dates, missing_px)
        sentinel2 = np.delete(sentinel2, missing_px, axis = 0)
        if clm is not None:
            clm = np.delete(clm, missing_px, axis = 0)

    # Otherwise... set the missing values to the median value.
    sentinel2 = interpolation.interpolate_missing_vals(sentinel2)
    if make_shadow:
        time1 = time()
        # Bounding box passed to remove_missed_clouds to mask 
        # out non-urban areas from the false positive cloud removal
        cloudshad, fcps = cloud_removal.remove_missed_clouds(sentinel2, dem, bbx)

        if clm is not None:
            clm[fcps] = 0.
            cloudshad = np.maximum(cloudshad, clm)

        interp = cloud_removal.id_areas_to_interp(
            sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
        )

        if verbose:
            print(f"INTERP: {100*np.mean(interp == 1, axis = (1, 2))}%")
        # In order to properly normalize band values to gapfill cloudy areas
        # We need 2% of the image to be non-cloudy
        # So that we can identify PIFs with at least 1000 px
        # Images deleted here will get propogated in the resegmentation
        # So it should not cause boundary artifacts in the final product.
        to_remove = np.argwhere(np.mean(interp == 1, axis = (1, 2)) > 0.98).flatten()
        if len(to_remove) > 0:
            #print(f"Deleting {to_remove}")
            clouds = np.delete(clouds, to_remove, axis = 0)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
            interp = np.delete(interp, to_remove, axis = 0)
            cloudshad, fcps = cloud_removal.remove_missed_clouds(sentinel2, dem, bbx)
            if clm is not None:
                clm = np.delete(clm, to_remove, axis = 0)
                cloudshad = np.maximum(cloudshad, clm)

            interp = cloud_removal.id_areas_to_interp(
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps)

        to_remove = np.argwhere(np.mean(interp == 1, axis = (1, 2)) > 0.98).flatten()
        if len(to_remove) > 0:
            #print(f"Deleting {to_remove}")
            clouds = np.delete(clouds, to_remove, axis = 0)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
            interp = np.delete(interp, to_remove, axis = 0)
            cloudshad, fcps = cloud_removal.remove_missed_clouds(sentinel2, dem, bbx)
            if clm is not None:
                clm = np.delete(clm, to_remove, axis = 0)
                cloudshad = np.maximum(cloudshad, clm)

        sentinel2, interp = cloud_removal.remove_cloud_and_shadows(
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps, wsize = 8, step = 8, thresh = 4)

        if verbose:
            time2 = time()
            print(f"Cloud/shadow interp:{np.around(time2 - time1, 1)} seconds")
            print(f"{100*np.sum(interp > 0.0, axis = (1, 2))/(interp.shape[1] * interp.shape[2])}%")
            print("Cloud/shad", np.mean(cloudshad, axis = (1, 2)))
    
    else:
        interp = np.zeros(
            (sentinel2.shape[0], sentinel2.shape[1], sentinel2.shape[2]), dtype = np.float32)
        cloudshad = np.zeros(
            (sentinel2.shape[0], sentinel2.shape[1], sentinel2.shape[2]), dtype = np.float32)

    dem = dem / 90
    sentinel2 = np.clip(sentinel2, 0, 1)

    # switch from monthly to annual median
    s1 = np.median(s1, axis = 0)
    s2 = np.median(sentinel2, axis = 0)
    
    # removing return of image_dates, interp, cloudshad as not used
    return s2, s1, dem


## Step 3: Combine raw data into a sample for input into the model 

def make_sample(dem: np.array, s1: np.array, s2: np.array, tml_feats: np.array):
    
    ''' 
    Takes processed data, defines dimensions for the sample, then 
    combines dem, s1, s2 and features into a single array with shape (x, x, 78)
    '''

    # define number of features in the sample
    n_feats = 1 + s1.shape[-1] + s2.shape[-1] + tml_feats.shape[-1] 

    # Create the empty array using shape of inputs
    sample = np.empty((dem.shape[0], dem.shape[1], n_feats))
    
    # populate empty array with each feature
    sample[..., 0] = dem
    sample[..., 1:3] = s1
    sample[..., 3:13] = s2
    sample[..., 13:] = tml_feats

    # save dims for future use
    arr_dims = (sample.shape[0], sample.shape[1])

    return sample, arr_dims

def make_sample_nofeats(dem: np.array, s1: np.array, s2: np.array):
    
    ''' 
    Takes processed data, defines dimensions for the sample and then 
    combines dem, s1, s2 into a single array with shape (x, x, 13). 
    TML features are excluded from the sample - No transfer learning. 
    '''
    
    # define the last dimension of the array
    n_feats = 1 + s1.shape[-1] + s2.shape[-1] 

    sample = np.empty((dem.shape[0], dem.shape[1], n_feats))
    
    # populate empty array with each feature
    sample[..., 0] = dem
    sample[..., 1:3] = s1
    sample[..., 3:13] = s2
    
    # save dims for future use
    arr_dims = (sample.shape[0], sample.shape[1])
    
    return sample, arr_dims

# Step 4: reshape and scale the sample

def reshape_and_scale_manual(v_train_data: str, unseen: np.array, verbose: bool = False):

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
    
    # if verbose:
    #     print(f"The data has been scaled. Min {start_min} -> {end_min}, Max {start_max} -> {end_max},")

    # now reshape
    unseen_reshaped = np.reshape(unseen, (np.prod(unseen.shape[:-1]), unseen.shape[-1]))

    return unseen_reshaped


def reshape_no_scaling(unseen: np.array, verbose: bool = False):

    ''' 
    Do not apply scaling and only reshape the unseen data.
    '''

    # now reshape
    unseen_reshaped = np.reshape(unseen, (np.prod(unseen.shape[:-1]), unseen.shape[-1]))

    if verbose:
        print(unseen_reshaped.shape)

    return unseen_reshaped


# Step 5: import classification model, run predictions

def predict_classification(arr: np.array, model: str, no_data_flag: np.array, no_tree_flag: np.array, sample_dims: tuple):

    '''
    Import pretrained model and run predictions on arr.
    If using a regression model, multiply results by 100
    to get probability 0-100. Reshape array to permit writing to tif.
    '''

    with open(f'models/{model}.pkl', 'rb') as file:  
        model_pretrained = pickle.load(file)
    
    preds = model_pretrained.predict(arr)
    
    if 'rfr' in model:
        preds = preds * 100

    reshaped_preds = preds.reshape(sample_dims[0], sample_dims[1])

    # apply no data and no tree flag to predictions
    # to clean up noise
    # reshaped_preds[no_data_flag] = 255.
    # reshaped_preds[no_tree_flag] = 0.

    return reshaped_preds


# Step 6: Write predictions for that tile to a tif -- eventually this will be a separate script?

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


# Execute steps
@timer
def execute(country: str, model: str, verbose: bool, feats: bool, feature_select: list):
    '''
    Executes all preprocessing and modeling steps in the pipeline
    according to the supplied model and country.
    '''
    local_dir = 'tmp/' + country

    tiles_to_process = download_tile_ids(country, aak, ask)
    tile_count = len(tiles_to_process)
    counter = 0

    # right now this will just process n tiles
    for tile_idx in tiles_to_process[8:11]:
        print(f'Processing tile: {tile_idx}')
        counter += 1
        successful = download_raw_tile(tile_idx, local_dir, aak, ask)

        if successful:
            validate.input_dtype_and_dimensions(tile_idx, local_dir)
            validate.feats_range(tile_idx, local_dir)
            bbx = make_bbox(country, tile_idx)
            s2_proc, s1_proc, dem_proc = process_tile(tile_idx, local_dir, bbx, verbose)
            validate.output_dtype_and_dimensions(s1_proc, s2_proc, dem_proc)

            # feats option will be removed in the future
            if feats:
                tml_feats, no_data_flag, no_tree_flag = process_tml_feats(tile_idx, local_dir, feats, feature_select)
                validate.tmlfeats_dtype_and_dimensions(tml_feats, feature_select)
                sample, sample_dims = make_sample(dem_proc, s1_proc, s2_proc, tml_feats)
                unseen_ss = reshape_no_scaling(sample, verbose)
                #unseen_ss = reshape_and_scale_manual('v11', sample, verbose)
    
            else:
                sample, sample_dims = make_sample_nofeats(dem_proc, s1_proc, s2_proc)
                #unseen_ss = reshape_and_scale_manual('v10', sample, verbose)
            
            validate.model_inputs(unseen_ss)
            preds = predict_classification(unseen_ss, model, no_data_flag, no_tree_flag, sample_dims)
            #validate.classification_scores(preds)
            write_tif(preds, bbx, tile_idx, country, 'preds')
            #remove_folder(tile_idx, local_dir)
        
        else:
            print(f'Raw data for {tile_idx} does not exist on s3.')
        
        if counter %5 == 0:
            print(f'{counter}/{tile_count} tiles processed...')
    
    # for now mosaic and upload to s3 bucket
    mosaic.mosaic_tif(country, model, compile_from='csv')
    #mosaic.upload_mosaic(country, model, aak, ask)
    
    return None


if __name__ == '__main__':
   
    import argparse
    parser = argparse.ArgumentParser()
    print("Argument List:", str(sys.argv))

    parser.add_argument('--country', dest='country', type=str)
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--verbose', dest='verbose', default=False, type=bool) 
    parser.add_argument('--feats', dest='feats', default=True, type=bool) 
    parser.add_argument('--feature_select', dest='feature_select', nargs='*', type=int) 


    args = parser.parse_args()
    
    execute(args.country, args.model, args.verbose, args.feats, args.feature_select)