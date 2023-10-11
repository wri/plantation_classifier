#!/usr/bin/env python

import numpy as np
from skimage.feature import graycomatrix, graycoprops
import hickle as hkl
import itertools
import functools
from time import time, strftime
from datetime import datetime

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

#@timer
def extract_texture(arr, properties_list = ['dissimilarity', 'correlation', 'homogeneity', 'contrast']):
    
    '''
    Given a s2 array, calculates a 5x5 sliding window
    by moving along axis 0 and then axis 1. Calculates the GLCM property 
    for a given window of the input array (28 x 28).
    Removes border information and returns the concatenated 14x14 arrays
    as a single output (14, 14, num properties)

    Bands must be calculated in this order: blue, green, red, nir
    Texture must be calculatd in this order:dissimilarity, correlation, homogeneity, contrast 
    
    '''
    # create windows - x.shape is (24, 24, 5, 5)
    windows = np.lib.stride_tricks.sliding_window_view(arr, (5,5), axis=(0,1))
    
    # hold all of the texture arrays
    texture_arr = np.zeros(shape=(14, 14, len(properties_list)), dtype=np.float32)
    index = 0
    
    # for every texture property
    for prop in properties_list:
        start = time()
        
        output = np.zeros((windows.shape[0], windows.shape[1]), dtype=np.float32)
        
        # for every item in range of 0-24
        for i, l in itertools.product(np.arange(windows.shape[0]), np.arange(windows.shape[1])):
            output[i, l] = glcm(windows[i, l, :, :], prop)
            
        # now slice out border information to get array from (28, 28) to (14, 14)
        border_x = (output.shape[0] - 14) // 2
        border_y = (output.shape[1] - 14) // 2
        cropped = output[border_x:-border_x, border_y:-border_y]
        cropped = cropped[..., np.newaxis]
        
        texture_arr[..., index:index+1] = cropped
        index += 1
        end = time()
        #print(f"Finished {prop} in {np.around(end - start, 1)} seconds.")
    
    return texture_arr

@timer
def deply_extract_texture(arr, properties_list):
    
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
    
    # print(f'arr {arr.shape} is padded to {padded.shape}')
    # print(f'windows shape {windows.shape}')
    # print(f'output arr shape: {texture_arr.shape}')

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