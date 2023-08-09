#!/usr/bin/env python

import os
import boto3
import botocore
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import hickle as hkl
import itertools
import functools
from time import time, strftime
from datetime import datetime

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

    s3_path = f'2020/ard/{str(x)}/{str(y)}/'
    s3_path_feats = f'2020/raw/{str(x)}/{str(y)}/raw/feats/'
    local_path = f'tmp/{country}/{str(x)}/{str(y)}/'

    # check if ARD folder has been downloaded
    ard_check = os.path.exists(local_path + 'ard/')

    if not ard_check:
        print(f"Downloading ARD for {(x, y)}")

        s3 = boto3.resource('s3',
                            aws_access_key_id=aws_access_key, 
                            aws_secret_access_key=aws_secret_key)
        bucket = s3.Bucket('tof-output')

        # this will download whatever is in the ard folder 
        for obj in bucket.objects.filter(Prefix=s3_path):

            ard_target = os.path.join(local_path + 'ard/', os.path.relpath(obj.key, s3_path))
            print(f'target download path: {ard_target}')

            if not os.path.exists(os.path.dirname(ard_target)):
                os.makedirs(os.path.dirname(ard_target))
            if obj.key[-1] == '/':
                continue
            try:
                bucket.download_file(obj.key, ard_target)

            # if the tiles do not exist on s3, catch the error and return False
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False
    
    ard = hkl.load(f'{local_path}ard/{str(x)}X{str(y)}_ard.hkl')
    return ard, True

def glcm(img, prop):
    '''
    Properties of a GLCM. 

    gray-level co-occurrence matrix: a histogram of co-occurring grayscale 
    values at a given offset over an image. The 4D output array is a GLCM
    histogram: levels x levels x number of distances x number of angles.  
        - distance: indicates the distance to move (1 = 1 pixel)
        - angle: indicates which direction to move (up, down, left, right)
        - level: indicates the number of gray-level pixels counted (typically 256 for 8-bit image)

    gray co-props: Calculates the texture properties of a GLCM. 
    Computes a feature of a GLCM to serve as a compact summary of the matrix. 
    The properties are computed as follows: contrast, dissimilarity, 
    homogeneity, energy, correlation, ASM. 

    '''

    # define params
    dist = [1] 
    angl = [0]
    lvl = 256 

    glcm = graycomatrix(img, distances=dist, angles=angl, levels=lvl)    
    glcm_props = graycoprops(glcm, prop)[0,0] # can add [0,0] to return...

    return glcm_props

@timer
def extract_texture(arr, properties_list):
    
    '''
    Given a s2 array, calculates a 5x5 sliding window on a padded array
    by moving along axis 0 and then axis 1. Calculates the GLCM property 
    for a given window of the input array (28x28).
    Concatenates the texture arrays as a single output with shape
    (618, 614, num properties)

    Bands must be calculated in this order: blue, green, red, nir
    Texture must be calculatd in this order:dissimilarity, correlation, homogeneity, contrast 
    
    '''
    # pad the arr to (622, 618)
    padded = np.pad(arr, ((2,2)), 'reflect')

    # create windows (614, 610, 5, 5)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5,5), axis=(0,1))
    
    # output arr is (618, 614, len(properties)
    texture_arr = np.zeros(shape=(arr.shape[0], arr.shape[1], len(properties_list)), dtype=np.float32)
    index = 0

    # for every texture property
    for prop in properties_list:
        start = time()
        output = np.zeros((windows.shape[0], windows.shape[1]), dtype=np.float32)
        
        # for every item in range of 0-610
        for i, l in itertools.product(np.arange(windows.shape[0]), np.arange(windows.shape[1])):
            output[i, l] = glcm(windows[i, l, :, :], prop)
        
        # now clip the output to align with original dims
        output = output[:arr.shape[0], :arr.shape[1]]
        output = output[..., np.newaxis]
        texture_arr[..., index:index+1] = output
        index += 1
        end = time()
        print(f"Finished {prop} in {np.around(end - start, 1)} seconds.")

    return texture_arr

def create_texture_array(tile_idx: tuple, location: list):
        
    # prepare inputs
    x = tile_idx[0]
    y = tile_idx[1]
    folder = f'tmp/{country}/{str(x)}/{str(y)}/'
    tile_str = f'{str(x)}X{str(y)}Y'
    ard = hkl.load(f'{folder}ard/{str(x)}X{str(y)}_ard.hkl') # note this file name is missing y
    s2 = ard[..., 0:10]
    s2 = img_as_ubyte(s2)
    assert s2.dtype == np.uint8, print(s2.dtype)
    
    txt = np.zeros((s2.shape[0], s2.shape[1], 16), dtype=np.float32)
        
    # FOR REGRESSION
    blue = s2[..., 0]
    green = s2[..., 1]
    red = s2[..., 2]
    nir = s2[..., 3]
    print('Calculating select GLCM textures for blue band...')
    txt[..., 0:4] = slow_txt.deply_extract_texture(blue, ['dissimilarity', 'correlation', 'homogeneity','contrast'])
    print('Calculating select GLCM textures for green band...')
    txt[..., 4:8] = slow_txt.deply_extract_texture(green, ['dissimilarity', 'correlation', 'homogeneity','contrast'])
    print('Calculating select GLCM textures for red band...')
    txt[..., 8:12] = slow_txt.deply_extract_texture(red, ['dissimilarity', 'correlation', 'homogeneity','contrast'])
    print('Calculating select GLCM textures for nir band...')
    txt[..., 12:16] = slow_txt.deply_extract_texture(nir, ['dissimilarity', 'correlation', 'homogeneity','contrast'])

    # save glcm texture properties in case
    np.save(f'{folder}raw/feats/{tile_str}_txt_lgr.npy', txt)

    # validate outputs and upload to s3
    # outpath will be the new filename
    suffix = f'{location[0]}_{model}_{date}.tif'
    txt_filepath = f'tmp/{location[0]}/preds/mosaic/{suffix}'

    s3 = boto3.resource('s3',
                        aws_access_key_id=aws_access_key, 
                        aws_secret_access_key=aws_secret_key)
    
    print(f'Uploading {mosaic_filepath} to s3.')

    s3.meta.client.upload_file(mosaic_filepath, 
                              'restoration-monitoring', 
                              'plantation-mapping/data/samples/' + suffix)

    return None