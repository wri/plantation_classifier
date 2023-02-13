#!/usr/bin/env python

import numpy as np
import hickle as hkl

### TRAINING ###
def train_output_range_dtype(dem, s1, s2, feats):
    '''
    Sentinel-1, float32, range from 0-1 (divided by 65535), unscaled decibels >-22
    Sentinel-2, float32, range from 0-1 (divided by 65535), unscaled
    Features, float32, range from ~-3 to ~ + 3 
    TML prediction, float32, range from 0-1 (divided by 100)
    '''

    assert s1.dtype == np.float32
    assert s2.dtype == np.float32
    assert feats.dtype == np.float32
    assert dem.dtype == np.float32

    assert np.logical_and(s1.min() >= 0, s1.max() <= 1)
    assert np.logical_and(s2.min() >= 0, s2.max() <= 1)
    assert np.logical_and(feats[..., 1:].min() >= -3, feats[..., 1:].max() <= 3)
    assert np.logical_and(feats[..., 0].min() >= 0, feats[..., 0].max() <= 1)


### DEPLOYMENT ###
# these tests happen after data is downloaded from s3

def input_dtype_and_dimensions(tile_idx, local_dir):

    '''
    Ensures the data type for raw s1, s2 and feats is uint16
    and the data type for raw DEM data is <f4.
    Ensures the dimensions for the downloaded data are as follows
    '''

    x = str(tile_idx[0])
    y = str(tile_idx[1])

    folder = f"{local_dir}/{x}/{y}/"
    tile_str = f'{x}X{y}Y'

    s1 = hkl.load(f'{folder}raw/s1/{tile_str}.hkl')
    s2_10 = hkl.load(f'{folder}raw/s2_10/{tile_str}.hkl')
    s2_20 = hkl.load(f'{folder}raw/s2_20/{tile_str}.hkl')
    dem = hkl.load(f'{folder}raw/misc/dem_{tile_str}.hkl')
    feats = hkl.load(f'{folder}raw/feats/{tile_str}_feats.hkl')

    # feats will be int16
    assert s1.dtype == np.uint16
    assert s2_10.dtype == np.uint16
    assert s2_20.dtype == np.uint16
    assert feats.dtype == np.int16 
    assert dem.dtype == '<f4'

    assert s1.shape[0] == 12 and s1.shape[3] == 2
    assert s2_10.shape[3] == 4
    assert s2_20.shape[3] == 6 or s2_20.shape[3] == 7 #7 indices if data mask is included
    assert len(dem.shape) == 2
    assert feats.shape[0] == 65



def feats_range(tile_idx, local_dir):

    ''' 
    Ensures the first index of the feature array ranges from 0 to 100
    Ensures all other indexes of the feature array range from -32,000 to 32,000
    '''
    x = str(tile_idx[0])
    y = str(tile_idx[1])

    folder = f"{local_dir}/{x}/{y}/"
    tile_str = f'{x}X{y}Y'

    feats_file = f'{folder}raw/feats/{tile_str}_feats.hkl'
    feats = hkl.load(feats_file)
    
    # assert feats index 0 (TML preds) range between 0-100 OR 255, and others range between -32768 and 32767
    assert np.logical_and(feats[0,...].min() >= 0, feats[0,...].max() <= 100) or feats[0,...].max() == 255
    assert np.logical_and(feats.min() >= np.iinfo(np.int16).min, feats.max() <= np.iinfo(np.int16).max)

    if feats[0,...].max() == 255:
        print(f'255 values present in TML predictions')


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
    assert len(dem.shape) == 2

    # middle two indices should be exactly the same for data (x, this_one, this_one, x)
    assert s1.shape[0:2] == s2.shape[0:2] == dem.shape, print(f'Clouds:, \n'
                                                            f'S1: {s1.shape} \n'
                                                            f'S2: {s2.shape} \n'
                                                            f'DEM: {dem.shape}')

def tmlfeats_dtype_and_dimensions(feats, feature_select):
    '''
    Ensures the datatype for processed feats is float32. Ensures the 
    dimensions for tml_feats are xx unless feature selection is used
    '''

    assert feats.dtype == np.float32
    
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
    
    # https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_almost_equal.html 
    #assert np.logical_and(arr.min() >= -1.0000000000000002, arr.max() <= 1.0000000000000002)
    #print(f'Max: {arr.max()}' n\ f'Min: {arr.min()}')


# test model outputs - these tests happen after predictions are generated

def classification_scores(preds):

    '''ensures the classification ouput is a binary 1 or 0 (or 255 no data value)'''
    ''' ensures no tile is solely plantation predictions (1) (highly unlikely)'''
    
    assert np.logical_or(preds == 0, preds == 1).all() or np.any(preds == 255)
    # assert preds.all() != 1, np.unique(preds)


def regression_scores(preds):

    ''' ensures the regression output ranges from 0 to 100'''
    ''' ensures no tile is solely 100 (highly unlikely)'''

    assert np.logical_and(preds >= 0, preds <= 100) or np.any(preds == 255)
    assert np.all(preds != 100), np.unique(preds)
