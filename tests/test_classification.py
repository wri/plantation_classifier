import numpy as np
import hickle as hkl

# first shot at some unit tests


# test data inputs - these tests happen after data is downloaded
# have to figure out workflow here

folder = f'{tile_idx[0]}X{tile_idx[1]}Y'
s1 = f'{folder}raw/s1/{tile_idx}.hkl'
s2_10 = f'{folder}raw/s2_10/{tile_idx}.hkl'
s2_20 = f'{folder}raw/s2_20/{tile_idx}.hkl'
dem = f'{folder}raw/misc/dem_{tile_idx}.hkl'
feats = f'{folder}raw/feats/{tile_idx}_feats.hkl'

def test_input_dtype_is_uint16():

    for i in [s1, s2_10, s2_20, dem, feats]:
        assert isinstance(hkl.load(i), np.uint16) # or <f4??

def test_input_dimensions():

    assert s1.shape[0] == 12 and s1.shape[3] == 2
    assert s2_10.shape[0] == 8 and s2_10.shape[3] == 4
    assert s2_20.shape[0] == 8 and s2_20.shape[3] == 7
    assert len(dem.shape) == 2
    assert feats.shape[0] == 65

def test_tree_cover_prediction_range():

    assert feats[0,...].min() >= 0
    assert feats[0,...].max() <= 100

def test_tree_cover_feats_range():

    for band in range(1, feats.shape[0]):
        assert feats[band, ...].min() >= -32000
        assert feats[band, ...].max() <= 32000


# test pre-processing - these tests happen after pre processing, before modeling

def test_output_dtype_is_float32():

    # how to grab processed data which is not explicitly saved
    for i in [s1, s2_10, s2_20, feats]:
        assert isinstance(hkl.load(i), np.float32)


def test_output_dimensions():

    assert s1.shape[3] == 2
    assert s2.shape[3] == 10
    assert len(dem.shape) == 2
    assert feats.shape[2] == 65


# test model outputs

def test_classification_scores():
    
    assert preds == 0 or preds == 1
    assert np.all(preds != 1)

def test_regression_scores():

    assert preds >= 0 and preds <= 100 
    assert np.all(preds != 100)

# separate test file for testing functions