#! /usr/bin/env python3

import numpy as np


def convert_to_db(x: np.ndarray, min_db: int) -> np.ndarray:
    """
    Converts Sentinel 1 unitless backscatter coefficient
    to decible with a min_db lower threshold

    Background: S1 is required to be processed to equivalent backscatter coefficient
    images in decibels (dB) scale. This backscatter coefficient represents the target
    backscattering area (radar cross-section) per unit ground area.
    It is required to be converted into dB as it can vary by several orders of magnitude.
    It measures whether the surface backscatters from the incident microwave radiation
    are preferentially away from the SAR sensor dB < 0) or towards the SAR sensor dB > 0).

    Parameters:
        x (np.ndarray): unitless backscatter (T, X, Y, B) array
        min_db (int): integer from -50 to 0
            (-22 db is the lower limit of sensitivity for s1)

    Returns:
        x (np.ndarray): db backscatter (T, X, Y, B) array

    (T, X, Y, B): time, x dimension, y dimension, band
    """
    # converts the array to decibel
    x = 10 * np.log10(x + 1 / 65535)
    x[x < -min_db] = -min_db
    x = (x + min_db) / min_db

    # return array clipped to values between 0-1
    x = np.clip(x, 0, 1)

    return x


def convert_float32(arr: np.ndarray) -> np.ndarray:
    # checks for floating datatype, if not converts to float32
    if not isinstance(arr.flat[0], np.floating):
        assert np.max(arr) > 1
        arr = arr.astype(np.float32) / 65535
        assert np.max(arr) < 1

    assert arr.dtype == np.float32

    return arr


def reshape_arr(arr):
    """
    Reshapes a 4D array (X) to 2D and a 3D array (y) to 1D for input into a
    machine learning model.

    Parameters:
    - arr (array-like): Input array to be reshaped. For X, it is assumed to have
    shape (plots, 14, 14, n_feats), and for y, it is assumed to have shape (plots, 14, 14).

    Returns:
    - reshaped (array-like): Reshaped array with dimensions suitable for machine learning model input.
    """
    if len(arr.shape) == 4:
        reshaped = np.reshape(arr, (np.prod(arr.shape[:-1]), arr.shape[-1]))
    else:
        reshaped = np.reshape(arr, (np.prod(arr.shape[:])))
    return reshaped
