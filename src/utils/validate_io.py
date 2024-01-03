#!/usr/bin/env python

import numpy as np
import hickle as hkl
import gc

### TRAINING ###
# these tests happen after preprocessing the raw training data

def fast_glcm_input(img):
    '''
    check that the input band meets the following criteria
    s2 should be shape (28, 28, 10) and dtype uint8
    ranging between 0 - 255
    '''

    assert len(img.shape) == 3
    assert img.dtype == np.uint8
    assert np.logical_and(img.min() >= 0, img.max() <= 255)


def fast_glcm_output(txt):
    '''
    check that the output of fast_glcm contains 4 texture
    properties for 4 bands (16 total) and is dtype float32
    '''
    assert txt.shape == (14, 14, 16)
    assert txt.dtype == np.float32


def train_output_range_dtype(s2, dem, s1, feats, feature_select):
    '''
    Sentinel-1, float32, range from 0-1 (divided by 65535), unscaled decibels >-22
    Sentinel-2, float32, range from 0-1 (divided by 65535), unscaled
    TML Features feats[..., 1:], float32, range from ~-3 to ~ + 3 
    TML prediction feats[..., 0], float32, range from 0-1 (deploy feats will be 0-100)
    '''

    assert s1.dtype == np.float32
    assert s2.dtype == np.float32
    assert dem.dtype == np.float32
    assert feats.dtype == np.float32

    assert np.logical_and(s1.min() >= 0, s1.max() <= 1), print(s1.min(), s1.max())
    assert np.logical_and(s2.min() >= 0, s2.max() <= 1), print(s2.min(), s2.max())

    # if there is no feature selection, assert feats meet logic
    # this only checks ttc feats and not txt feats
    if len(feature_select) < 1:
        assert np.logical_and(feats[..., 1:65].min() >= -32.768, feats[..., 1:65].max() <= 32.768), print(feats[..., 1:65].min(), feats[..., 1:65].max())
        assert np.logical_and(feats[..., 0].min() >= 0, feats[..., 0].max() <= 1), print(feats[..., 0].min(), feats[..., 0].max())
   
    # TODO: enable validation when feature selection used

    
### DEPLOYMENT ###
# these tests happen after data is downloaded from s3

def input_dtype_and_dimensions(tile_idx, country):

    '''
    Ensures the data type for raw s1, s2 and feats is uint16
    and the data type for raw DEM data is <f4.
    Ensures the dimensions for the downloaded data are as follows
    '''

    x = str(tile_idx[0])
    y = str(tile_idx[1])

    folder = f"tmp/{country}/{x}/{y}/"
    tile_str = f'{x}X{y}Y'

    s1 = hkl.load(f'{folder}raw/s1/{tile_str}.hkl')
    s2_10 = hkl.load(f'{folder}raw/s2_10/{tile_str}.hkl')
    s2_20 = hkl.load(f'{folder}raw/s2_20/{tile_str}.hkl')
    dem = hkl.load(f'{folder}raw/misc/dem_{tile_str}.hkl')
    ttc_feats = hkl.load(f'{folder}raw/feats/{tile_str}_feats.hkl')

    # feats will be int16
    assert s1.dtype == np.uint16
    assert s2_10.dtype == np.uint16
    assert s2_20.dtype == np.uint16
    assert ttc_feats.dtype == np.int16 
    assert dem.dtype == '<f4'

    assert s1.shape[0] == 12 and s1.shape[3] == 2
    assert s2_10.shape[3] == 4
    assert s2_20.shape[3] == 6 or s2_20.shape[3] == 7 #7 indices if data mask is included
    assert len(dem.shape) == 2
    assert ttc_feats.shape[0] == 65

    del s1, s2_10, s2_20, dem, ttc_feats


def input_ard(tile_idx, country):

    '''
    ARD will be between 0-1 in all cases except for SWIR bands, which are 
    downloaded at 40-m and super-resolved to 10m. This super-resolution 
    process is technically not bounded to the 0-1 range, so could in rare 
    occasions predict a negative value. It appears that here some water 
    pixels get assigned a negative value. Therefore the assertion counts the
    number of pixels that are out of range and only fails if that exceeds 10,000.
    '''
    
    x = str(tile_idx[0])
    y = str(tile_idx[1])

    folder = f"tmp/{country}/{x}/{y}/"
    tile_str = f'{x}X{y}Y'
    print(f'{folder}ard/{tile_str}_ard.hkl')
    ard = hkl.load(f'{folder}ard/{tile_str}_ard.hkl')

    assert ard.dtype == np.float32
    assert ard.shape[2] == 13

    pixel_counter = 0
    if not np.logical_and(ard.min() >= 0, ard.max() <= 1):
        pixel_counter += 1
    assert pixel_counter < 10000, print(f'{pixel_counter} pixels have negative values.')

    del ard


def feats_range(tile_idx, country):

    ''' 
    Ensures the first index of the feature array ranges from 0 to 100
    Ensures all other indexes of the feature array range from -32,000 to 32,000
    '''
    x = str(tile_idx[0])
    y = str(tile_idx[1])

    folder = f"tmp/{country}/{x}/{y}/"
    tile_str = f'{x}X{y}Y'

    feats_file = f'{folder}raw/feats/{tile_str}_feats.hkl'
    feats = hkl.load(feats_file)
    
    # assert feats index 0 (TML preds) range between 0-100 OR 255, and others range between -32768 and 32767
    assert np.logical_and(feats[0,...].min() >= 0, feats[0,...].max() <= 100) or feats[0,...].max() == 255
    assert np.logical_and(feats.min() >= np.iinfo(np.int16).min, feats.max() <= np.iinfo(np.int16).max)

    if feats[0,...].max() == 255:
        print(f'255 values present in TML predictions')
    
    del feats_file, feats


# test pre-processing - these tests happen after pre processing


def output_dtype_and_dimensions(s1, s2, dem):

    '''
    Ensures the datatype for all processed data (output of process_tile())
    is float32. Ensures the dimensions for the processed data are as follows:
    '''
    # as long as the workflow includes feats, otherwise feats will be a str
    assert s1.dtype == np.float32
    assert s2.dtype == np.float32
    assert dem.dtype == np.float32

    assert s1.shape[2] == 2
    assert s2.shape[2] == 10
    assert len(dem.shape) == 2, print(dem.shape)

    # ensure array is not 0s, would indicate weird behavior
    assert len(np.unique(s1)) > 1

    # middle two indices should be exactly the same for all data (x, this_one, this_one, x)
    assert s1.shape[0:2] == s2.shape[0:2] == dem.shape, print(f'S1: {s1.shape} \n'
                                                              f'S2: {s2.shape} \n'
                                                              f'DEM: {dem.shape}')


def tmlfeats_dtype_and_dimensions(dem, feats, feature_select):
    '''
    Ensures the datatype for processed feats is float32. Ensures the 
    dimensions for tml_feats are (x, x, 65) unless feature selection is used
    Takes in dem to assert dims match
    '''

    assert feats.dtype == np.float32
    assert feats.shape[0:2] == dem.shape, print(f'WARNING. Feats: {feats.shape} DEM: {dem.shape}')
    
    if len(feature_select) > 0:
        assert feats.shape[2] == len(feature_select)
    else:
        assert feats.shape[2] == 65


def model_inputs(arr):
    '''
    Ensures the range of all scaled data is between -1-1
    it is reshaped to 2-D array, and contains no NaN or INF values.
    '''
    assert len(arr.shape) == 2
    assert np.isfinite(arr).all()
    
    # assert that the min and max are almost equal to 15 decimals
    # assert_almost_equal(arr.min(), -1.0000000000000002, decimal=15)
    # assert_almost_equal(arr.max(), 1.0000000000000002, decimal=15)
    #assert np.logical_and(arr.min() >= -1.0000000000000002, arr.max() <= 1.0000000000000002)


# test model outputs - these tests happen after predictions are generated

def model_outputs(arr, type):

    '''
    Ensures the classification ouput is 0, 1, 2, 3 or 255 
    and ensures no tile is solely monoculture predictions (1) 
    which is highly unlikely.

    Ensures the regression output is between 0-100 or 255
    and ensures no tile is solely 100% which is highly unlikely.
    '''

    # chain together multiple logical_or calls with reduce
    if type == 'classifier':
        assert np.logical_or.reduce((arr == 0, arr == 1, arr == 2, arr == 3)).all() or np.any(arr == 255)
        assert arr.all() != 1

    elif type == 'regressor':
        assert np.logical_and(arr >= 0, arr <= 100).all() or np.any(arr == 255), print(np.unique(arr))
        assert np.all(arr != 100)

    
## validate texture array 

def texture_output_dims(arr):
    '''
    Confirm texture arr shape, dtype and count of properties

    '''
    assert len(arr.shape) == 3
    assert arr.shape[-1] == 16
    assert arr.dtype == np.float32

def texture_output_range(arr, prop):
    '''
    Then check texture calculations fall within appropriate ranges
    
    Correlation Range = -1 to 1
    Homogeneity Range = 0 to 1
    Dissimilarity Range = min >= 0
    Contrast Range = min >= 0
    '''

    if prop == 'dissimilarity':
        assert arr.min() >= 0.0, print(arr.min())
    
    elif prop == 'correlation':
        assert arr.min() >= -1.0 and arr.max() <= 1.0, print(arr.min(), arr.max())
    
    elif prop == 'homogeneity':
        assert arr.min() >= 0.0 and arr.max() <= 1.0, print(arr.min(), arr.max())

    elif prop == 'contrast':
        assert arr.min() >= 0.0, print(arr.min())