import numpy as np
import hickle as hkl
import pytest

# first shot at some unit tests
# when would this be triggered? After download of each tile?

##### GROUP 1
# Starting point - data inputs have been downloaded
# Now need to test assumptions about their format

def setup(tile_idx):

    '''imports the data for xx tile to test'''

    folder = f'{tile_idx[0]}X{tile_idx[1]}Y'
    s1 = f'{folder}raw/s1/{tile_idx}.hkl'
    s2_10 = f'{folder}raw/s2_10/{tile_idx}.hkl'
    s2_20 = f'{folder}raw/s2_20/{tile_idx}.hkl'
    dem = f'{folder}raw/misc/dem_{tile_idx}.hkl'
    feats = f'{folder}raw/feats/{tile_idx}_feats.hkl'

    return s1, s2_10, s2_20, dem, feats


def test_input_dtype_is_uint16(s1, s2_10, s2_20, feats):

    '''ensures the data type for all downloaded data is uint16'''

    for i in [s1, s2_10, s2_20, feats]:
        arr = hkl.load(i)
        assert arr.dtype == np.uint16


def test_dem_dtype_is_f4(dem):

    '''ensures the data type for DEM data is <f4'''

    arr = hkl.load(dem)
    assert arr.dtype == '<f4'



def test_input_dimensions(s1, s2_10, s2_20, dem, feats):

    '''ensure the dimensions for the downloaded data are as follows'''

    assert s1.shape[0] == 12 and s1.shape[3] == 2
    assert s2_10.shape[0] == 8 and s2_10.shape[3] == 4
    assert s2_20.shape[0] == 8 and s2_20.shape[3] == 7
    assert len(dem.shape) == 2
    assert feats.shape[0] == 65


def test_tree_cover_prediction_range(feats):

    ''' ensures the first index of the feature array ranges from 0 to 100'''

    assert feats[0,...].min() >= 0
    assert feats[0,...].max() <= 100


def test_tree_cover_feats_range(feats):

    ''' ensures all other indexes of the feature array range from -32,000 to 32,000'''

    for band in range(1, feats.shape[0]):
        assert feats[band, ...].min() >= -32000
        assert feats[band, ...].max() <= 32000


##### GROUP 2
# test pre-processing - these tests happen after pre processing
# before modeling
# how to grab processed data which is not explicitly saved in the pipeline??

def test_output_dtype_is_float32(s1, s2, dem, feats):

    '''ensures the datatype for all processed data is float32'''

    for i in [s1, s2, dem, feats]:
        assert isinstance(hkl.load(i), np.float32)


def test_output_dimensions(s1, s2, dem, feats):

    '''ensures the dimensions for the processed data are as follows'''

    assert s1.shape[3] == 2
    assert s2.shape[3] == 10
    assert len(dem.shape) == 2
    assert feats.shape[2] == 65


##### GROUP 3
# test model outputs
# this would have to be triggered once the pipeline is completed

def test_classification_scores():

    '''ensures the classification ouput is a binary 1 or 0'''
    ''' ensures no tile is solely 1 (highly unlikely)'''
    
    assert preds == 0 or preds == 1
    assert np.all(preds != 1)

def test_regression_scores():

    ''' ensures the regression output ranges from 0 to 100'''
    ''' ensures no tile is solely 100 (highly unlikely)'''

    assert preds >= 0 and preds <= 100 
    assert np.all(preds != 100)


##### GROUP 4
# separate test file for testing functions are performing as expected

def test_