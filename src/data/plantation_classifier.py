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
from datetime import datetime, timezone
from scipy import ndimage
from skimage.util import img_as_ubyte
import gc
import copy


## import other scripts
import sys
sys.path.append('src/')
import src.utils.interpolation as interpolation
import src.utils.cloud_removal as cloud_removal
import src.utils.mosaic as mosaic
import src.features.validate_io as validate
import src.features.slow_glcm as slow_txt
import src.features.fast_glcm as fast_txt

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


def download_s3(s3_folder: str, local_dir: str, aws_access_key: str, aws_secret_key: str):
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


def download_raw_tile(tile_idx: tuple, country: str, access_key: str, secret_key: str, update_feats: bool) -> None:
    '''
    If data is not present locally, downloads raw data (clouds, DEM and 
    image dates, s1 and s2 (10 and 20m bands)) for the specified tile from s3. 
    Not free to run downloads to local - use sparingly.
    '''

    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
    
    # state local path and s3
    local_raw = f'tmp/{country}/{str(x)}/{str(y)}/'
    # doesnt work s3_raw = f'/2020/raw/{str(x)}/{str(y)}/raw/'
    s3_raw = f'2020/raw/{str(x)}/{str(y)}/'
    
    # check if s1 folder exists locally
    folder_check = os.path.exists(local_raw + f"raw/s1/{str(x)}X{str(y)}Y.hkl")

    if folder_check:
        print('Raw data exists locally.')
    
    # if feats folder doesn't exist locally, download raw tile
    if not folder_check:
        print(f"Downloading data for {(x, y)}")
        try: 
            download_s3(s3_folder = s3_raw,
                        local_dir = local_raw,
                        aws_access_key = access_key,
                        aws_secret_key = secret_key)

        # if the tiles do not exist on s3, catch the error and return False
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            
    if update_feats:
        
        # Create an S3 client and define bucket
        s3_client = boto3.client('s3')
        bucket_name = 'tof-output'

        # try these first
        s3_feats = f'2020/raw/{str(x)}/{str(y)}/raw/feats/{str(x)}X{str(y)}Y_feats.hkl'
        local_feats = f'tmp/{country}/{str(x)}/{str(y)}/raw/feats/{str(x)}X{str(y)}Y_feats.hkl'

        # Get the metadata of the S3 object
        response = s3_client.head_object(Bucket=bucket_name, 
                                         Key=s3_feats)

        # Compare the LastModified timestamps
        s3_last_modified = response['LastModified'].replace(tzinfo=timezone.utc)
        local_last_modified = os.path.getmtime(local_feats)
        local_last_modified = datetime.fromtimestamp(local_last_modified, tz=timezone.utc)

        if s3_last_modified > local_last_modified:
            print('Updating TTC features with most recent version.')
            # Remote version is newer, download the file
            s3_client.download_file(bucket_name, 
                                    s3_feats, 
                                    local_feats)
        else:
            # Local version is up to date, no need to download
            print("TTC features are up to date.")
        
    return True


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
    bbx_df = pd.read_csv(f"data/{country}.csv")

    # this will remove quotes around x and y tile indexes (not needed for all countries)
    # data['X_tile'] = data['X_tile'].str.extract('(\d+)', expand=False)
    # data['X_tile'] = pd.to_numeric(data['X_tile'])
    # data['Y_tile'] = data['Y_tile'].str.extract('(\d+)', expand=False)
    # data['Y_tile'] = pd.to_numeric(data['Y_tile'])

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

def process_tile(tile_idx: tuple, country: str, bbx: list, verbose: bool = False, make_shadow: bool = True) -> np.ndarray:
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
    
    x = tile_idx[0]
    y = tile_idx[1]
            
    folder = f"tmp/{country}/{str(x)}/{str(y)}/"
    tile_str = f'{str(x)}X{str(y)}Y'

    clouds_file = f'{folder}raw/clouds/clouds_{tile_str}.hkl'
    cloud_mask_file = f'{folder}raw/clouds/cloudmask_{tile_str}.hkl'
    s1_file = f'{folder}raw/s1/{tile_str}.hkl'
    s2_10_file = f'{folder}raw/s2_10/{tile_str}.hkl'
    s2_20_file = f'{folder}raw/s2_20/{tile_str}.hkl'
    s2_dates_file = f'{folder}raw/misc/s2_dates_{tile_str}.hkl'
    dem_file = f'{folder}raw/misc/dem_{tile_str}.hkl'
    
    # load clouds
    clouds = hkl.load(clouds_file)
    if os.path.exists(cloud_mask_file):
        # These are the S2Cloudless / Sen2Cor masks
        clm = hkl.load(cloud_mask_file).repeat(2, axis = 1).repeat(2, axis = 2)
    else:
        clm = None

    # load s1
    s1 = hkl.load(s1_file)
    s1 = np.float32(s1) / 65535
    s1[..., -1] = convert_to_db(s1[..., -1], 22)
    s1[..., -2] = convert_to_db(s1[..., -2], 22)
    
    # load s2
    s2_10 = to_float32(hkl.load(s2_10_file))
    s2_20 = to_float32(hkl.load(s2_20_file))
    
    # slice off last index (no data mask) if present
    if s2_20.shape[3] == 7:
        s2_20 = s2_20[..., :6]
        #print(f's2_20 data mask removed.')

    # load dem
    dem = hkl.load(dem_file)
    dem = median_filter(dem, size = 5)    
    
    # Ensure arrays are the same dims (asserts s2_10 is 2x s2_20)
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

    # combine s2_10 and s2_10 bands into one array
    # bilinearly upsample 20m bands to 10m for superresolution - superresolution is not actually happening
    # BUT shouldn't actually change map quality much
    sentinel2 = np.zeros((s2_10.shape[0], width, height, 10), np.float32)
    sentinel2[..., :4] = s2_10

    # a for loop is faster than trying to vectorize it here! 
    for band in range(4):
        for step in range(sentinel2.shape[0]):
            sentinel2[step, ..., band + 4] = resize(s2_20[step,..., band], (width, height), 1)

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
    image_dates = hkl.load(s2_dates_file)
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

    # John to confirm 
    dem = dem / 90
    sentinel2 = np.clip(sentinel2, 0, 1)

    # switch from monthly to annual median
    s1 = np.median(s1, axis = 0, overwrite_input=True)
    s2 = np.median(sentinel2, axis = 0, overwrite_input=True)

    del s2_10, s2_20, sentinel2, image_dates, clouds, missing_px, interp

    # temporarily save to file
    # hkl.dump(s2, '../tmp/s2_ghana.hkl', mode='w')
    # hkl.dump(s1, '../tmp/s1_ghana.hkl', mode='w')
    # hkl.dump(dem, '../tmp/dem_ghana.hkl', mode='w')
    # np.save('../tmp/s2_cr.npy', s2)
    # np.save('../tmp/s1_cr.npy', s1)
    # np.save('../tmp/dem_cr.npy', dem)

    # removing return of image_dates, interp, cloudshad as not used
    return s2, s1, dem

def process_ttc(tile_idx: tuple, country: str, incl_feats: bool, feature_select:list) -> np.ndarray:
    '''
    Transforms the feats with shape (65, x, x) extracted from the TML model 
    (in temp/raw/tile_feats..) to processed data structure
        - scale tree prediction (feats[0]) between 0-1 to match the training
          pipeline 
        - roll the axis to adjust shape
        - swap high and low level feats to match training pipeline
        - filter to selected feats if feature_select param > 0
        - creates no data and no tree flags for masking predictions
    '''

    x = tile_idx[0]
    y = tile_idx[1]

    folder = f"tmp/{country}/{str(x)}/{str(y)}/"
    tile_str = f'{str(x)}X{str(y)}Y'

    # load and prep features here
    if incl_feats:
        feats_file = f'{folder}raw/feats/{tile_str}_feats.hkl'
        feats_raw = hkl.load(feats_file).astype(np.float32)
    
        # adjust TML predictions feats[0] to match training data (0-1)
        # adjust shape by rolling axis 2x (65, 614, 618) ->  (618, 614, 65) 
        # feats used for deply are multiplyed by 1000 before saving
        feats_raw[0, ...] = feats_raw[0, ...] / 100 
        feats_raw[1:, ...] = feats_raw[1:, ...] / 1000  
        feats_rolled = np.rollaxis(feats_raw, 0, 3)
        feats_rolled = np.rollaxis(feats_rolled, 0, 2)
        
        # now switch the feats
        feats_ = copy.deepcopy(feats_rolled)

        high_feats = [np.arange(1,33)]
        low_feats = [np.arange(33,65)]

        feats_[:, :, [low_feats]] = feats_rolled[:, :, [high_feats]]
        feats_[:, :, [high_feats]] = feats_rolled[:, :, [low_feats]]

        # create no data and no tree flag (boolean mask)
        # where TML probability is 255 or 0, pass along to preds
        # note that the feats shape is (x, x, 65)
        no_data_flag = feats_[...,0] == 255.
        no_tree_flag = feats_[...,0] <= 0.1

        # if only using select feats, filter to those
        if len(feature_select) > 0:
            feats_ = np.squeeze(feats_[:, :, [feature_select]])

    # in case we are doing a no feats analysis
    # remove this else statement once pipeline updated to feat only 
    else:
        feats_ = []

    del feats_raw, feats_rolled, high_feats, low_feats

    return feats_, no_data_flag, no_tree_flag

def process_full_glcm_slow(s2):
    
    '''
    Takes in a (x, x, 10) s2 array and performs texture analysis
    on all four bands. Returns an comb output containing the 4
    texture analyses for the four bands.
    '''
    s2 = img_as_ubyte(s2)
    assert s2.dtype == np.uint8, print(s2.dtype)
    
    blue = s2[..., 0]
    green = s2[..., 1]
    red = s2[..., 2]
    nir = s2[..., 3]
    output = np.zeros((14, 14, 16), dtype=np.float32)
    
    print('Calculating GLCM textures for blue band...')
    output[..., 0:4] = slow_txt.extract_texture(blue)
    print('Calculating GLCM textures for green band...')
    output[..., 4:8] = slow_txt.extract_texture(green)
    print('Calculating GLCM textures for red band...')
    output[..., 8:12] = slow_txt.extract_texture(red)
    print('Calculating GLCM textures for nir band...')
    output[..., 12:16] = slow_txt.extract_texture(nir)

    return output.astype(np.float32)

@timer
def process_feats_fast(tile_idx: tuple, country: str, incl_feats: bool, feature_select:list, s2) -> np.ndarray:
    '''
    Transforms the feats with shape (65, x, x) extracted from the TML model 
    (in temp/raw/tile_feats..) to processed data structure
        - scale tree prediction (feats[0]) between 0-1 to match the training
          pipeline 
        - roll the axis to adjust shape
        - swap high and low level feats to match training pipeline
        - filter to selected feats if feature_select param > 0
        - creates no data and no tree flags for masking predictions

    Import txt features if upload_txt is True, otherwise calculates txt properties with
    fast_glcm implementation.

    Combines ttc features THEN txt features in output array. Applies feature selection to 
    only ttc features
    '''
    
    x = tile_idx[0]
    y = tile_idx[1]

    folder = f"tmp/{country}/{str(x)}/{str(y)}/"
    tile_str = f'{str(x)}X{str(y)}Y'
    
    # output shape will match s2 array num of ttc features + 8 txt for v19
    n_feats = len(feature_select) + 6
    output = np.zeros((s2.shape[0], s2.shape[1], n_feats), dtype=np.float32)
    print(f'output shape is {output.shape}')

    # load and prep features here
    if incl_feats:
        feats_file = f'{folder}raw/feats/{tile_str}_feats.hkl'
        feats_raw = hkl.load(feats_file).astype(np.float32)
    
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
        # where TTC is 255 or <0.1 
        no_data_flag = ttc[...,0] == 255.
        no_tree_flag = ttc[...,0] <= 0.1

        # apply feature selection to ttc feats 
        if len(feature_select) > 0:
            ttc = np.squeeze(ttc[:, :, [feature_select]])
            print(f'ttc shape is {ttc.shape}')

    # in case we are doing a no feats analysis
    # remove this else statement once pipeline updated to feat only 
    else:
        ttc = []

    # import txt features if available, otherwise calc them
    if os.path.exists(f'{folder}raw/feats/{tile_str}_txt_blah.npy'):
        print('Importing texture features.')
        txt = np.load(f'{folder}raw/feats/{tile_str}_txt_blah.npy')

    else:
        print('Calculating texture features.')
        
        # convert img as type float32 to uint8
        img = img_as_ubyte(s2)
        green = img[..., 1]
        red = img[..., 2]
        nir = img[..., 3]
        
        # required order is blue, green, red, nir
        # dissimilarity, correlation, homogeneity, contrast 
        print('Calculating select GLCM textures for green band...')
        green_txt = fast_txt.extract_texture(green, ['dissimilarity', 'correlation', 'homogeneity', 'contrast'], pipeline='deply')
        print('Calculating select GLCM textures for red band...')
        red_txt = fast_txt.extract_texture(red, ['contrast'], pipeline='deply')
        print('Calculating select GLCM textures for nir band...')
        nir_txt = fast_txt.extract_texture(nir, ['dissimilarity', 'correlation', 'contrast'], pipeline='deply')

        # should be like this
        # [..., 0:4] = green
        # [..., 4:7] = red
        # [..., 7:] = nir
        property_count = green_txt.shape[-1] + red_txt.shape[-1] + nir_txt.shape[-1]
        txt = np.zeros((img.shape[0], img.shape[1], property_count), dtype=np.float32)
        print(green_txt.shape, red_txt.shape, nir_txt.shape)
        print(f'output arr shape: {txt.shape}')

        txt[..., 0:green_txt.shape[-1]] = green_txt 
        txt[..., green_txt.shape[-1]: green_txt.shape[-1] + red_txt.shape[-1]] = red_txt
        txt[..., - nir_txt.shape[-1]:] = nir_txt

        # save glcm texture properties in case
        np.save(f'{folder}raw/feats/{tile_str}_txt_newtest.npy', txt)

    # now combine ttc feats with txt feats
    output[..., :ttc.shape[-1]] = ttc
    output[..., ttc.shape[-1]:] = txt

    del feats_raw, feats_rolled, high_feats, low_feats, ttc, txt

    return output, no_data_flag, no_tree_flag


@timer
def process_feats_slow(tile_idx: tuple, country: str, incl_feats: bool, feature_select:list, s2) -> np.ndarray:
    '''
    Transforms the feats with shape (65, x, x) extracted from the TML model 
    (in temp/raw/tile_feats..) to processed data structure
        - scale tree prediction (feats[0]) between 0-1 to match the training
          pipeline 
        - roll the axis to adjust shape
        - swap high and low level feats to match training pipeline
        - filter to selected feats if feature_select param > 0
        - creates no data and no tree flags for masking predictions
    '''
    
    x = tile_idx[0]
    y = tile_idx[1]

    folder = f"tmp/{country}/{str(x)}/{str(y)}/"
    tile_str = f'{str(x)}X{str(y)}Y'
    
    # output shape will match s2 array ttc feats and 5 txt feats
    txt_feats = 8
    n_feats = len(feature_select) + txt_feats 
    output = np.zeros((s2.shape[0], s2.shape[1], n_feats), dtype=np.float32)


    # load and prep features here
    if incl_feats:
        feats_file = f'{folder}raw/feats/{tile_str}_feats.hkl'
        feats_raw = hkl.load(feats_file).astype(np.float32)
    
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
        no_tree_flag = ttc[...,0] <= 0.1

        # apply feature selection to ttc feats 
        if len(feature_select) > 0:
            ttc = np.squeeze(ttc[:, :, [feature_select]])

    # in case we are doing a no feats analysis
    # remove this else statement once pipeline updated to feat only 
    else:
        ttc = []

    # import txt features if available, otherwise calc them
    if os.path.exists(f'{folder}raw/feats/{tile_str}_txtv19.npy'):
        print('Importing texture features.')
        txt = np.load(f'{folder}raw/feats/{tile_str}_txtv19.npy')

    else:
        s2 = img_as_ubyte(s2)
        assert s2.dtype == np.uint8, print(s2.dtype)
        
        # dissimilarity, correlation, homogeneity, contrast 
        txt = np.zeros((s2.shape[0], s2.shape[1], txt_feats), dtype=np.float32)
        green = s2[..., 1]
        red = s2[..., 2]
        nir = s2[..., 3]
        print('Calculating select GLCM textures for green band...')
        txt[..., 0:1] = slow_txt.deply_extract_texture(green, ['contrast'])
        print('Calculating select GLCM textures for red band...')
        txt[..., 1:2] = slow_txt.deply_extract_texture(red, ['contrast'])
        print('Calculating select GLCM textures for nir band...')
        txt[..., 2:] = slow_txt.deply_extract_texture(nir, ['dissimilarity', 'correlation', 'contrast'])

        # save glcm texture properties in case
        np.save(f'{folder}raw/feats/{tile_str}_txt_blah.npy', txt)

    output[..., :ttc.shape[-1]] = ttc
    output[..., ttc.shape[-1]:] = txt

    del feats_raw, feats_rolled, high_feats, low_feats, ttc

    return output, no_data_flag, no_tree_flag

## Step 3: Combine raw data into a sample for input into the model 

def make_sample(dem: np.array, s1: np.array, s2: np.array, feats: np.array):
    
    ''' 
    Takes processed data, defines dimensions for the sample, then 
    combines dem, s1, s2 and features into a single array with shape (x, x, len(features))
    Dimensions
    dem (618, 614)
    s1 (618, 614, 2)
    s2 (618, 614, 10)
    feats (618, 614, 40)
    sample (618, 614, 53)
    '''

    # define number of features in the sample (add one for dem)
    n_feats = 1 + s1.shape[-1] + s2.shape[-1] + feats.shape[-1] 
    # ard = hkl.load(f'tmp/ghana/1667/1077/ard/1667X1077_ard.hkl')
    # s2 = ard[..., 0:10]

    # Create the empty array using shape of inputs
    sample = np.zeros((dem.shape[0], dem.shape[1], n_feats), dtype=np.float32)

    # populate empty array with each feature
    sample[..., 0] = dem
    sample[..., 1:3] = s1
    sample[..., 3:13] = s2
    sample[..., 13:] = feats

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


def reshape(arr: np.array, verbose: bool = False):

    ''' 
    Do not apply scaling and only reshape the unseen data.
    '''

    arr_reshaped = np.reshape(arr, (np.prod(arr.shape[:-1]), arr.shape[-1]))

    if verbose:
        print(arr_reshaped.shape)

    return arr_reshaped


# Step 5: import classification model, run predictions

def predict_classification(arr: np.array, pretrained_model: str, sample_dims: tuple):

    '''
    Import pretrained model and run predictions on arr.
    If using a regression model, multiply results by 100
    to get probability 0-100. Reshape array to permit writing to tif.
    '''
    
    preds = pretrained_model.predict(arr)
    
    # TODO: update peipeline for regression
    # if 'rfr' in model:
    #     preds = preds * 100

    return preds.reshape(sample_dims[0], sample_dims[1])

def remove_small_patches(arr, thresh):
    
    '''
    Label features in an array using ndimage.label() and count 
    pixels for each lavel. If the count doesn't meet provided
    threshold, make the label 0. Return an updated array

    (option to add a 3x3 structure which considers features connected even 
    if they touch diagonally - probably dnt want this)
    
    '''

    # creates arr where each unique feature (non zero value) has a unique label
    # num features are the number of connected patches
    labeled_array, num_features = ndimage.label(arr)

    # get pixel count for each label
    label_size = [(labeled_array == label).sum() for label in range(num_features + 1)]

    for label,size in enumerate(label_size):
        if size < thresh:
            arr[labeled_array == label] = 0
    
    return arr

def post_process_tile(arr: np.array, feature_select: list, no_data_flag: np.array, no_tree_flag: np.array):

    '''
    Applies the no data and no tree flag *if* TTC tree cover predictions are used
    in feature selection. The NN produces a float32 continuous prediction.

    Performs a connected component analysis to remove positive predictions 
    where the connected pixel count is < thresh. 
    '''

    # FLAG: requires feature selection
    if 0 in feature_select:
        arr[no_data_flag] = 255.
        arr[no_tree_flag] = 0.

    postprocess_mono = remove_small_patches(arr == 1, thresh = 20)
    postprocess_af = remove_small_patches(arr == 2, thresh = 15)
    
    # multiplying by boolean will turn every False into 0 
    # and keep every True as the original label
    arr[arr == 1] *= postprocess_mono[arr == 1]
    arr[arr == 2] *= postprocess_af[arr == 2]
  
    del postprocess_af, postprocess_mono

    return arr

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
    
    del arr, new_dataset

    return None

def remove_folder(tile_idx: tuple, country: str):
    '''
    Deletes temporary raw data files in path_to_tile/raw/*
    after predictions are written to file
    '''
    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
  
    path_to_tile = f'tmp/{country}/{str(x)}/{str(y)}/'

    # remove every folder/file in raw/
    for folder in glob(path_to_tile + "raw/*/"):
        for file in os.listdir(folder):
            _file = folder + file
            os.remove(_file)
        
    return None

def execute_per_tile(tile_idx: tuple, location: list, model, verbose: bool, incl_feats: bool, feature_select: list):
    
    print(f'Processing tile: {tile_idx}')
    successful = download_raw_tile(tile_idx, location[0], aak, ask, update_feats=False)

    if successful:
        validate.input_dtype_and_dimensions(tile_idx, location[0])
        validate.feats_range(tile_idx, location[0])
        bbx = make_bbox(location[1], tile_idx)
        s2_proc, s1_proc, dem_proc = process_tile(tile_idx, location[0], bbx, verbose)
        validate.output_dtype_and_dimensions(s1_proc, s2_proc, dem_proc)

        # feats option will be removed in the future
        if incl_feats:
            feats, no_data_flag, no_tree_flag = process_feats_slow(tile_idx, location[0], incl_feats, feature_select, s2_proc)
            #tml_feats, no_data_flag, no_tree_flag = process_ttc(tile_idx, local_dir, incl_feats, feature_select)
            #validate.tmlfeats_dtype_and_dimensions(dem_proc, feats, feature_select)
            #txt_feats = process_txt_feats_select(s2_proc)
            # sample, sample_dims = make_sample(dem_proc, s1_proc, s2_proc, tml_feats, txt_feats)
            sample, sample_dims = make_sample(dem_proc, s1_proc, s2_proc, feats)
            sample_ss = reshape(sample, verbose)
            #sample_ss = reshape_and_scale_manual('v17', sample, verbose)

        else:
            sample, sample_dims = make_sample_nofeats(dem_proc, s1_proc, s2_proc)
            sample_ss = reshape(sample, verbose)
            #sample_ss = reshape_and_scale_manual('v10', sample, verbose)
        
        validate.model_inputs(sample_ss)
        preds = predict_classification(sample_ss, model, sample_dims)
        preds_final = post_process_tile(preds, feature_select, no_data_flag, no_tree_flag)
        validate.model_outputs(preds_final, 'classifier')
        
        write_tif(preds_final, bbx, tile_idx, location[0], 'preds')
        
        # clean up memory
        #remove_folder(tile_idx, location[0])
        del bbx, s2_proc, s1_proc, dem_proc, feats, no_data_flag, no_tree_flag, sample, sample_ss, preds, preds_final
    
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
    
    #execute(args.country, args.model, args.verbose, args.feats, args.feature_select)

    # specify tiles HERE
    tiles_to_process = download_tile_ids(args.location, aak, ask)[1:2]
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
    