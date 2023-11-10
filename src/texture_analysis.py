#!/usr/bin/env python

import yaml
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
from skimage.util import img_as_ubyte
import sys
from glob import glob
sys.path.append('src/')

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

        if len(objs) > 0:
            print(f"The s3 resource s3://{bucket.name}/{s3_file} will be downloaded.")
            bucket.download_file(s3_file, dest_file)
            
    database = pd.read_csv(dest_file)

    # create a list of tiles 
    tiles = database[['X_tile', 'Y_tile']].to_records(index=False)

    return tiles

def file_exists(tile_idx: tuple, aws_access_key: str, aws_secret_key: str):
    '''
    Checks if the texture array has already been created 
    and saved on s3. If not, proceeds to calculation.
    '''

    x = tile_idx[0]
    y = tile_idx[1]
    bucket = 'tof-output'
    key = f'2020/raw/{str(x)}/{str(y)}/raw/feats/{str(x)}X{str(y)}Y_txt.npy'

    s3 = boto3.client('s3',
                aws_access_key_id=aws_access_key, 
                aws_secret_access_key=aws_secret_key)

    try:
        s3.head_object(Bucket=bucket, Key=key)
        print(f'Texture file already exists for {x, y}')

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f'No texture file exists for {x, y}')
            return False

    return True

def download_ard(tile_idx: tuple, location: list, aws_access_key: str, aws_secret_key: str):
    ''' 
    If ARD folder is not present locally,
    Download the ard file for the specified tile (does
    not download entire folder contents).
    '''
    x = tile_idx[0]
    y = tile_idx[1]
    tile_str = f'{str(x)}X{str(y)}Y'

    s3_file = f'2020/ard/{str(x)}/{str(y)}/{tile_str}_ard.hkl'
    local_dir = f'tmp/{location[0]}/{str(x)}/{str(y)}/ard/'
    local_file = f'{local_dir}{tile_str}_ard.hkl'

    s3 = boto3.resource('s3',
                        aws_access_key_id=aws_access_key, 
                        aws_secret_access_key=aws_secret_key)

    bucket = s3.Bucket('tof-output')
    if not os.path.exists(os.path.dirname(local_dir)):
        os.makedirs(os.path.dirname(local_dir))
    try:
        bucket.download_file(s3_file, local_file)

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f'Error downloading ARD file for {x, y}')
            return False

    return True

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
def extract_txt(arr, properties_list):
    
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
        start = datetime.now()
        output = np.zeros((windows.shape[0], windows.shape[1]), dtype=np.float32)
        
        # for every item in range of 0-610
        for i, l in itertools.product(np.arange(windows.shape[0]), np.arange(windows.shape[1])):
            output[i, l] = glcm(windows[i, l, :, :], prop)
        
        # now clip the output to align with original dims
        output = output[:arr.shape[0], :arr.shape[1]]
        output = output[..., np.newaxis]
        texture_arr[..., index:index+1] = output
        index += 1
        validate.texture_output_range(output, prop)
        end = datetime.now()
        print(f"Finished {prop} in {end - start}")

    return texture_arr

def create_txt_array(tile_idx: tuple, location: list, aws_access_key: str, aws_secret_key: str):
    
    # prepare inputs and create local folder
    x = tile_idx[0]
    y = tile_idx[1]
    folder = f'tmp/{location[0]}/{str(x)}/{str(y)}/'
    tile_str = f'{str(x)}X{str(y)}Y'
    if not os.path.exists(f'{folder}raw/feats/'):
        os.makedirs(f'{folder}raw/feats/')

    # prep s2 input
    ard = hkl.load(f'{folder}ard/{tile_str}_ard.hkl') 
    s2 = ard[..., 0:10]
    s2 = img_as_ubyte(s2)
    assert s2.dtype == np.uint8, print(s2.dtype)

    txt = np.zeros((s2.shape[0], s2.shape[1], 16), dtype=np.float32)
        
    blue = s2[..., 0]
    green = s2[..., 1]
    red = s2[..., 2]
    nir = s2[..., 3]
    print('Calculating select GLCM textures for blue band...')
    txt[..., 0:4] = extract_txt(blue, ['dissimilarity', 'correlation', 'homogeneity','contrast'])
    print('Calculating select GLCM textures for green band...')
    txt[..., 4:8] = extract_txt(green, ['dissimilarity', 'correlation', 'homogeneity','contrast'])
    print('Calculating select GLCM textures for red band...')
    txt[..., 8:12] = extract_txt(red, ['dissimilarity', 'correlation', 'homogeneity','contrast'])
    print('Calculating select GLCM textures for nir band...')
    txt[..., 12:16] = extract_txt(nir, ['dissimilarity', 'correlation', 'homogeneity','contrast'])

    # save and upload to s3
    validate.texture_output_dims(txt)
    np.save(f'{folder}raw/feats/{tile_str}_txt.npy', txt)
    s3_file = f'2020/raw/{str(x)}/{str(y)}/raw/feats/{tile_str}_txt.npy'
    print(f'Uploading {s3_file}')   
    client = boto3.client('s3',
                        aws_access_key_id=aws_access_key, 
                        aws_secret_access_key=aws_secret_key)
    
    client.upload_file(f'{folder}raw/feats/{tile_str}_txt.npy', "tof-output", s3_file)

    return None

def remove_folder(tile_idx: tuple, location: list):
    '''
    Deletes temporary files after uploaded to s3
    '''
    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
  
    path_to_tile = f'tmp/{location[0]}/{str(x)}/{str(y)}/'

    # remove every folder/file in raw/
    for folder in glob(path_to_tile + "raw/*/"):
        for file in os.listdir(folder):
            _file = folder + file
            print(f'Deleting {_file}')
            os.remove(_file)
        
    return None


if __name__ == '__main__':
   
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', dest='location', nargs='+', type=str)
    args = parser.parse_args()

    tiles_to_process = download_tile_ids(args.location, aak, ask)
    tile_count = len(tiles_to_process)
    counter = 0

    print('............................................')
    print(f'Processing {tile_count} tiles for {args.location[1], args.location[0]}.')
    print('............................................')

    for tile_idx in tiles_to_process:
        exists = file_exists(tile_idx, aak, ask)
        if not exists:
            print(f'Processing tile: {tile_idx}')
            successful = download_ard(tile_idx, args.location, aak, ask)
            if successful:
                create_txt_array(tile_idx, args.location, aak, ask)
                remove_folder(tile_idx, args.location)
                counter += 1
                if counter % 2 == 0:
                    print(f'{counter}/{tile_count} tiles processed...')
        else: # still add one if exists
            counter += 1