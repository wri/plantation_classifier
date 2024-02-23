## legacy code for importing raw dem, sentinel-1 
## and sentinel-2 data, as opposed to ARD
import hickle as hkl
import yaml
import pandas as pd
import numpy as np
import os
import utils.preprocessing as preprocess

def load_slope(idx, local_dir):
    """
    Slope is stored as a 32 x 32 float32 array with border information.
    Needs to be converted to 1 x 14 x 14 array to match the labels w/ new axis.
    """
    directory = f"{local_dir}train-slope/"

    # Remove axes of length one with .squeeze (32 x 32 x 1)
    x = np.load(directory + str(idx) + ".npy").squeeze()

    # slice out border information
    border_x = (x.shape[0] - 14) // 2
    border_y = (x.shape[1] - 14) // 2
    slope = x[border_x:-border_x, border_y:-border_y]

    # explicitly convert array to a row vector, adding axis (1 x 14 x 14)
    # slope = slope[np.newaxis]
    slope = slope[..., np.newaxis]

    return slope


def load_s1(idx, local_dir):
    """
    S1 is stored as a (12, 32, 32, 2) float64 array with border information.
    Needs to be converted from monthly mosaics to an annual median,
    and remove border information to match labels.
    Dtype needs to be converted to float32.
    """
    directory = f"{local_dir}train-s1/"

    # since s1 is a float64, can't use to_float32()
    s1 = hkl.load(directory + str(idx) + ".hkl")

    # convert to decible
    s1[..., -1] = preprocess.convert_to_db(s1[..., -1], 22)
    s1[..., -2] = preprocess.convert_to_db(s1[..., -2], 22)

    # get the median across flattened array
    if len(s1.shape) == 4:
        s1 = np.median(s1, axis=0, overwrite_input=True)

    # just convert (removed /65535) keep here
    s1 = np.float32(s1)

    # slice out border information (32, 32, 2) -> (14, 14, 2)
    border_x = (s1.shape[0] - 14) // 2
    border_y = (s1.shape[1] - 14) // 2
    s1 = s1[border_x:-border_x, border_y:-border_y]

    return s1


def load_s2(idx, local_dir):
    """
    S2 is stored as a (12, 28, 28, 11) uint16 array.
    Remove the last axis index - the date of the imagery.
    Convert monthly images to an annual median.
    Remove the border to correspond to the labels.
    Convert to float32.
    """
    directory = f"{local_dir}train-s2/"
    s2 = hkl.load(directory + str(idx) + ".hkl")

    # remove date of imagery (last axis)
    if s2.shape[-1] == 11:
        s2 = np.delete(s2, -1, -1)

    # checks for floating datatype, if not converts to float32
    # TODO check if this is the same func as in deply pipeline
    # if so consider moving to utils
    if not isinstance(s2.flat[0], np.floating):
        assert np.max(s2) > 1
        s2 = s2.astype(np.float32) / 65535
        assert np.max(s2) < 1

    # convert monthly images to annual median
    if len(s2.shape) == 4:
        s2 = np.median(s2, axis=0, overwrite_input=True)

    # slice out border information
    border_x = (s2.shape[0] - 14) // 2
    border_y = (s2.shape[1] - 14) // 2
    s2 = s2[border_x:-border_x, border_y:-border_y].astype(np.float32)

    return s2