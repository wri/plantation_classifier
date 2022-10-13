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
import confuse
import rasterio as rs
from osgeo import gdal
import time
from scipy.ndimage import median_filter
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

import sys
sys.path.append('../src/')
import interpolation
import cloud_removal
from prototype import prepare_data 


config = confuse.Configuration('plantation-classifier')
config.set_file('/Users/jessica.ertel/plantation_classifier/config.yaml')
aws_access_key = config['aws']['aws_access_key_id'].as_str()
aws_secret_key = config['aws']['aws_secret_access_key'].as_str()


## Step 1: Download raw data from s3

def download_folder(s3_folder: str, local_dir: str, apikey, apisecret):
    """
    Download the contents of the tof-output s3 folder directory
    into a local folder.

    Args:
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    
    s3 = boto3.resource('s3', aws_access_key_id=apikey, aws_secret_access_key=apisecret)
    bucket = s3.Bucket('tof-output')

    
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
        
    return None


def download_raw_tile(tile_idx: tuple, local_dir: str = "../tmp") -> None:
    
    '''
    Not free to run downloads to local - use sparingly.
    Downloads all files (clouds, misc: DEM and image dates, s1 and s2 (10 and 20m bands))
    for the specified tile.
    
    '''

    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
    
    # state local path and s3
    path_to_tile = f'{local_dir}/{str(x)}/{str(y)}/'
    s3_path_to_tile = f'2020/raw/{str(x)}/{str(y)}/'
    
    # check if the clouds folder already exists locally
    folder_to_check = os.path.exists(path_to_tile + "raw/clouds/")
    if folder_to_check:
        print('Exists locally.')
        return True
    
    # if the clouds folder doesn't exist locally, download raw tile
    # and return True
    if not folder_to_check:
        print(f"Downloading {s3_path_to_tile}")
        try: 
            s3 = boto3.resource('s3')
            s3.Object('tof-output', s3_path_to_tile + f'raw/s1/{str(x)}X{str(y)}Y.hkl').load()
            download_folder(s3_folder = s3_path_to_tile,
                            local_dir = path_to_tile,
                            apikey = aws_access_key,
                            apisecret = aws_secret_key)
            return True

        # if the tiles do not exist on s3, catch the error
        # and return False
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False


## Step 2: Create a cloud free composte

def make_bbox(database, tile_idx: tuple, expansion: int = 10) -> list:
    """
    Makes a (min_x, min_y, max_x, max_y) bounding box that
       is 2 * expansion 300 x 300 meter ESA LULC pixels

       Parameters:
            initial_bbx (list): [min_x, min_y, max_x, max_y]
            expansion (int): 1/2 number of 300m pixels to expand

       Returns:
            bbx (list): expanded [min_x, min_y, max_x, max_y]
    """
    data = pd.read_csv(f"../data/{database}.csv")

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
    return bbx_df, bbx

def convert_to_db(x: np.ndarray, min_db: int) -> np.ndarray:
    """ Converts Sentinel 1unitless backscatter coefficient
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
    Converts an int array to float32
    """

    print(f'The original max value is {np.max(array)}')
    if not isinstance(array.flat[0], np.floating):
        assert np.max(array) > 1
        array = np.float32(array) / 65535.
    assert np.max(array) <= 1
    assert array.dtype == np.float32
    return array

def adjust_shape(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Assures that the shape of arr is width x height
    Used to align 10, 20, 40, 160, 640 meter resolution Sentinel data
    """
    print(f"Input array shape: {arr.shape}")
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

def process_tile(x: int, y: int, data: pd.DataFrame, local_path: str, bbx: list, make_shadow: bool = True, verbose=False) -> np.ndarray:
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
    """
    
    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y
            
    folder = f"{local_path}/{str(x)}/{str(y)}/"
    tile_idx = f'{str(x)}X{str(y)}Y'
    
    clouds_file = f'{folder}raw/clouds/clouds_{tile_idx}.hkl'
    cloud_mask_file = f'{folder}raw/clouds/cloudmask_{tile_idx}.hkl'
    shadows_file = f'{folder}raw/clouds/shadows_{tile_idx}.hkl'
    s1_file = f'{folder}raw/s1/{tile_idx}.hkl'
    s1_dates_file = f'{folder}raw/misc/s1_dates_{tile_idx}.hkl'
    s2_10_file = f'{folder}raw/s2_10/{tile_idx}.hkl'
    s2_20_file = f'{folder}raw/s2_20/{tile_idx}.hkl'
    s2_dates_file = f'{folder}raw/misc/s2_dates_{tile_idx}.hkl'
    s2_file = f'{folder}raw/s2/{tile_idx}.hkl'
    clean_steps_file = f'{folder}raw/clouds/clean_steps_{tile_idx}.hkl'
    dem_file = f'{folder}raw/misc/dem_{tile_idx}.hkl'
    
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

    dem = hkl.load(dem_file)
    dem = median_filter(dem, size = 5)
    image_dates = hkl.load(s2_dates_file)
    
    # Ensure arrays are the same dims
    width = s2_20.shape[1] * 2
    height = s2_20.shape[2] * 2
    s1 = adjust_shape(s1, width, height)
    s2_10 = adjust_shape(s2_10, width, height)
    dem = adjust_shape(dem, width, height)

    if verbose:
        print(f'Clouds: {clouds.shape}, \n'
            f'S1: {s1.shape} \n'
            f'S2: {s2_10.shape}, {s2_20.shape} \n'
            f'DEM: {dem.shape}')

    # Deal with cases w/ only 1 image
    if len(s2_10.shape) == 3:
        s2_10 = s2_10[np.newaxis]
    if len(s2_20.shape) == 3:
        s2_20 = s2_20[np.newaxis]

    # bilinearly upsample 20m bands to 10m for superresolution
    sentinel2 = np.zeros((s2_10.shape[0], width, height, 10), np.float32)
    sentinel2[..., :4] = s2_10

    # a foor loop is faster than trying to vectorize it here! 
    for band in range(4):
        for step in range(sentinel2.shape[0]):
            sentinel2[step, ..., band + 4] = resize(
                s2_20[step,..., band], (width, height), 1
            )

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
                    np.int(np.floor(mid.shape[0] / 2)), 2,
                    np.int(np.floor(mid.shape[1] / 2)), 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, 1:, 1:, band + 4] = resize(mid, (width - 1, height - 1), 1)
                sentinel2[step, 0, :, band + 4] = mid_misaligned_x.repeat(2)
                sentinel2[step, :, 0, band + 4] = mid_misaligned_y.repeat(2)
            elif mid.shape[0] % 2 != 0:
                mid_misaligned = mid[0, :]
                mid = mid[1:].reshape(np.int(np.floor(mid.shape[0] / 2)), 2, mid.shape[1] // 2, 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, 1:, :, band + 4] = resize(mid, (width - 1, height), 1)
                sentinel2[step, 0, :, band + 4] = mid_misaligned.repeat(2)
            elif mid.shape[1] % 2 != 0:
                mid_misaligned = mid[:, 0]
                mid = mid[:, 1:]
                mid = mid.reshape(mid.shape[0] // 2, 2, np.int(np.floor(mid.shape[1] / 2)), 2)
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
        time1 = time.time()
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
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
            )

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
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps, wsize = 8, step = 8, thresh = 4
            )
        """
        time2 = time.time()
        print(f"Cloud/shadow interp:{np.around(time2 - time1, 1)} seconds")
        print(f"{100*np.sum(interp > 0.0, axis = (1, 2))/(interp.shape[1] * interp.shape[2])}%")
        #print("Cloud/shad", np.mean(cloudshad, axis = (1, 2)))
        """
    else:
        interp = np.zeros(
            (sentinel2.shape[0], sentinel2.shape[1], sentinel2.shape[2]), dtype = np.float32
        )
        cloudshad = np.zeros(
            (sentinel2.shape[0], sentinel2.shape[1], sentinel2.shape[2]), dtype = np.float32
        )

    dem = dem / 90
    sentinel2 = np.clip(sentinel2, 0, 1)
    sns.heatmap(sentinel2[0, ..., 0])

    return sentinel2, image_dates, interp, s1, dem, cloudshad

## PLACEHOLDER download feats (identify input and output )

## Step 3: Create a sample for input into the model -- right now using no TML features

def make_sample(slope, s1, s2, feats):
    
    ''' 
    Takes the output of process_tile() and calculates the monthly median
    Defines dimensions and then combines slope, s1, s2 and TML features from a plot
    into a sample with shape (x, x, 78)
    '''
    
    # define the last dimension of the array
    n_feats = 1 + s1.shape[-1] + s2.shape[-1] + feats.shape[-1]

    # Confirm if sample shape is a 2D array 
    sample = np.empty((slope.shape[0], slope.shape[1], n_feats))

    # switch from monthly to annual median
    s1 = np.median(s1, axis = 0)
    s2 = np.median(s2, axis = 0)
    
    # populate empty array with each feature
    sample[..., 0] = slope
    sample[..., 1:3] = s1
    sample[..., 3:13] = s2
    sample[..., 13:] = feats

    # save dims for future use
    arr_dims = (sample.shape[0], sample.shape[1])
    
    return sample, arr_dims

def make_sample_nofeats(slope, s1, s2):
    
    ''' 
    Takes the output of process_tile(), defines dimensions 
    and then combines slope, s1 and s2 features from a tile
    into a sample with shape (x, x, 13). 
    TML features are excluded from the sample - No transfer learning. 
    '''
    
    # define the last dimension of the array
    n_feats = 1 + s1.shape[-1] + s2.shape[-1] 
    print(f'features: {n_feats}')

    sample = np.empty((slope.shape[0], slope.shape[1], n_feats))
    print(f'sample shape: {sample.shape}')

    # convert monthly images to annual median
    s1 = np.median(s1, axis = 0) 
    s2 = np.median(s2, axis = 0)
    
    # populate empty array with each feature
    sample[..., 0] = slope
    sample[..., 1:3] = s1
    sample[..., 3:13] = s2
    
    # save dims for future use
    arr_dims = (sample.shape[0], sample.shape[1])
    
    return sample, arr_dims

# Step 4: reshape and scale the sample

def reshape_and_scale(v_train_data: list, unseen, verbose=False):
    
    '''
    V_training_data: list of training data version
    unseen: new sample 

    Takes in a tile (sample) with dimensions (x, x, 13) and 
    reshapes to (x, 13), then applies standardization.
    '''
    # prepare original training data for vectorizer
    # drop feats for now - UPDATE LATER
    X, y = prepare_data.create_xy((14,14), v_train_data, drop_prob=False, drop_feats=True, verbose=False)

    # train test split before reshaping to ensure plot is not mixed samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)

    # reshape arrays (only Xtrain and unseen)
    X_train_reshaped = np.reshape(X_train, (np.prod(X_train.shape[:-1]), X_train.shape[-1]))
    unseen_reshaped = np.reshape(unseen, (np.prod(unseen.shape[:-1]), unseen.shape[-1]))
    if verbose:
        print(f'Xtrain Original: {X_train.shape} Xtrain Reshaped: {X_train_reshaped.shape}')
        print(f'Unseen Original: {unseen.shape} Unseen Reshaped: {unseen_reshaped.shape}')

    # apply standardization on a copy
    X_train_ss = X_train_reshaped.copy()
    unseen_ss = unseen_reshaped.copy()

    scaler = StandardScaler()
    X_train_ss = scaler.fit_transform(X_train_ss)
    unseen_ss = scaler.transform(unseen_ss)
    if verbose:
        print(f'Scaled to {np.min(X_train_ss)}, {np.max(X_train_ss)}')
        print(f'Scaled to {np.min(unseen_ss)}, {np.max(unseen_ss)}')
    
    return unseen_ss


# Step 5: import classification model, run predictions

def predict_classification(arr, model, sample_dims):

    '''
    Import the reshaped and scaled data and model version,
    run predictions and output a numpy array of predictions per 6x6 km tile
    '''

    with open(f'../models/{model}.pkl', 'rb') as file:  
        model_pretrained = pickle.load(file)
    
    preds = model_pretrained.predict(arr)
    reshaped_preds = preds.reshape(sample_dims[0], sample_dims[1])
    #print(np.unique(reshaped_preds))

    return reshaped_preds


# Step 6: Write predictions for that tile to a tif 

def write_tif(arr: np.ndarray, bbx, tile_idx, country, suffix = "preds") -> str:
    
     # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
    out_folder = f'../tmp/{country}/preds/'
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
    new_dataset = rs.open(file, 'w', driver = 'GTiff',
                               height = arr.shape[0], width = arr.shape[1], count = 1,
                               dtype = "uint8",
                               compress = 'lzw',
                               crs = '+proj=longlat +datum=WGS84 +no_defs',
                               transform=transform)
    new_dataset.write(arr, 1)
    new_dataset.close()
    
    return None


# Execute steps

def execute(tile_idx, country, model):

    ## make the training data and model an input
    
    local_dir = '../tmp/' + country
    successful = download_raw_tile((tile_idx[0], tile_idx[1]), local_dir)
    if successful:
        bbx_df, bbx = make_bbox(country, (tile_idx[0], tile_idx[1]))
        s2_proc, image_dates, interp, s1_proc, slope_proc, cloudshad = process_tile(tile_idx[0], tile_idx[1], bbx_df, local_dir, bbx)
        sample, sample_dims = make_sample_nofeats(slope_proc, s1_proc, s2_proc)
        unseen_ss = reshape_and_scale(['v8'], sample)
        preds = predict_classification(unseen_ss, model, sample_dims)
        write_tif(preds, bbx, tile_idx, country, 'preds')
    else:
        print(f'Raw data for {tile_idx} does not exist.')
    return None


###

if __name__ == '__main__':
   
    import argparse
    parser = argparse.ArgumentParser()
    print("Argument List:", str(sys.argv))

    parser.add_argument('--tile_idx', dest='tile_idx', nargs='+', type=int)
    parser.add_argument('--country', dest='country', type=str)
    parser.add_argument('--model', dest='model', type=str)

    args = parser.parse_args()
    
    execute(args.tile_idx, args.country, args.model)