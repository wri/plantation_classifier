#!/usr/bin/env python

import numpy as np
from skimage.feature import graycomatrix, graycoprops
import hickle as hkl
import itertools



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
    angl = [0, np.pi/4, np.pi/2, 3*np.pi/4] 
    lvl = 256 
    
    glcm = graycomatrix(img, distances=dist, angles=angl, levels=lvl)    
    glcm_props = graycoprops(glcm, prop)[0,0] # can add [0,0] to return...
    
    return glcm_props


def extract_texture(arr):
    
    '''
    Given a s2 array, calculates a 5x5 sliding window
    by moving along axis 0 and then axis 1. Calculates the GLCM property 
    for a given window of the input array (24x24).
    Removes border information and returns the concatenated 14x14 arrays
    as a single output (14, 14, num properties)
    
    '''
    # create windows - x.shape is (24, 24, 5, 5)
    windows = np.lib.stride_tricks.sliding_window_view(arr, (5,5), axis=(0,1))
    
    # hold all of the texture arrays
    properties_list = ['dissimilarity', 'correlation', 'homogeneity', 'contrast']
    texture_arr = np.zeros(shape=(14, 14, len(properties_list)))
    index = 0
    
    # for every texture property
    for prop in properties_list:
        
        output = np.zeros((windows.shape[0], windows.shape[1]))
        
        # for every item in range of 0-24
        for i, l in itertools.product(np.arange(windows.shape[0]), np.arange(windows.shape[1])):
            output[i, l] = glcm(windows[i, l, :, :], prop)
            
        # now slice out border information to get array from (24, 24) to (14, 14)
        border_x = (output.shape[0] - 14) // 2
        border_y = (output.shape[1] - 14) // 2
        cropped = output[border_x:-border_x, border_y:-border_y]
        cropped = cropped[..., np.newaxis]
        
        texture_arr[..., index:index+1] = cropped
        index += 1
    
    return texture_arr