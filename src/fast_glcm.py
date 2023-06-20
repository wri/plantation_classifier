# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2
import functools
from datetime import datetime


### CODE ADAPTED FROM https://github.com/tzm030329/GLCM/blob/master/fast_glcm.py
### Credit to Taka Izumi

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

def fast_glcm(img, vmin=0, vmax=255, levels=256, kernel_size=5, distance=1.0, angle=0.0):
    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    kernel_size: int
        Patch size to calculate GLCM around the target pixel
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)

    Returns
    -------
    Grey-level co-occurrence matrix for each pixels
    shape = (levels, levels, h, w)
    '''

    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, levels+1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance*np.cos(np.deg2rad(angle))
    dy = distance*np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w,h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm

@timer
def fast_glcm_contrast(img, vmin=0, vmax=255, levels=256, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            cont += glcm[i,j] * (i-j)**2

    # now slice out border information to get array from (24, 24) to (14, 14)
    border_x = (cont.shape[0] - 14) // 2
    border_y = (cont.shape[1] - 14) // 2
    cont_crp = cont[border_x:-border_x, border_y:-border_y, np.newaxis]

    return cont_crp

@timer
def fast_glcm_dissimilarity(img, vmin=0, vmax=255, levels=256, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            diss += glcm[i,j] * np.abs(i-j)

    # now slice out border information to get array from (24, 24) to (14, 14)
    border_x = (diss.shape[0] - 14) // 2
    border_y = (diss.shape[1] - 14) // 2
    diss_crp = diss[border_x:-border_x, border_y:-border_y, np.newaxis]

    return diss_crp

@timer
def fast_glcm_homogeneity(img, vmin=0, vmax=255, levels=256, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm homogeneity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            homo += glcm[i,j] / (1.+(i-j)**2)

    # now slice out border information to get array from (24, 24) to (14, 14)
    border_x = (homo.shape[0] - 14) // 2
    border_y = (homo.shape[1] - 14) // 2
    homo_crp = homo[border_x:-border_x, border_y:-border_y, np.newaxis]

    return homo_crp

@timer
def fast_glcm_correlation(img, vmin=0, vmax=255, levels=256, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm correlation
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    corr = np.zeros((h,w), dtype=np.float32)

    # Create a 4D numpy array of integers that contains values from 0 to num_levels-1
    I = np.array(range(levels)).reshape((levels, 1, 1, 1))
    J = np.array(range(levels)).reshape((1, levels, 1, 1))
    
    # Calculate the difference between I and the sum of I * glcm along the 0 and 1 axis
    diff_i = I - np.sum(I * glcm, axis=(0, 1))
    diff_j = J - np.sum(J * glcm, axis=(0, 1))
    
    # calculate standard dev and covariance
    std_i = np.sqrt(np.sum(glcm * (diff_i) ** 2, axis=(0, 1)))
    std_j = np.sqrt(np.sum(glcm * (diff_j) ** 2, axis=(0, 1)))
    cov = np.sum(glcm * (diff_i * diff_j), axis=(0, 1))

    # handle the special case of standard deviations near zero
    mask_0 = std_i < 1e-15
    mask_0[std_j < 1e-15] = True
    corr[mask_0] = 1

    # handle the standard case
    mask_1 = ~mask_0
    corr[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])

    # now slice out border information to get array from (24, 24) to (14, 14)
    border_x = (corr.shape[0] - 14) // 2
    border_y = (corr.shape[1] - 14) // 2
    corr_crp = corr[border_x:-border_x, border_y:-border_y, np.newaxis]

    return corr_crp

def fast_glcm_train(img):

    blue = img[..., 0]
    green = img[..., 1]
    red = img[..., 2]
    nir = img[..., 3]
    output = np.zeros((14, 14, 16))

    output[...,0:1] = fast_glcm_dissimilarity(blue)
    output[...,1:2] = fast_glcm_correlation(blue)
    output[...,2:3] = fast_glcm_homogeneity(blue)
    output[...,3:4] = fast_glcm_contrast(blue)

    output[...,4:5] = fast_glcm_dissimilarity(green)
    output[...,5:6] = fast_glcm_correlation(green)
    output[...,6:7] = fast_glcm_homogeneity(green)
    output[...,7:8] = fast_glcm_contrast(green)

    output[...,8:9] = fast_glcm_dissimilarity(red)
    output[...,9:10] = fast_glcm_correlation(red)
    output[...,10:11] = fast_glcm_homogeneity(red)
    output[...,11:12] = fast_glcm_contrast(red)

    output[...,12:13] = fast_glcm_dissimilarity(nir)
    output[...,13:14] = fast_glcm_correlation(nir)
    output[...,14:15] = fast_glcm_homogeneity(nir)
    output[...,15:16] = fast_glcm_contrast(nir)

    return output.astype(np.float32)

def fast_glcm_deply(img):
    '''
    TODO: bands and texture properties shouldnt be hard coded
    so it's easier to change or select a specific band and 
    texture.
    '''

    blue = img[..., 0]
    green = img[..., 1]
    red = img[..., 2]
    nir = img[..., 3]
    output = np.zeros((14, 14, 8))

    # print('Calculating blue band texture properties...')
    # output[...,0:1] = fast_glcm_homogeneity(blue)
    # output[...,1:2] = fast_glcm_contrast(blue)

    print('Calculating green band texture properties...')
    output[...,0:1] = fast_glcm_dissimilarity(green)
    output[...,1:2] = fast_glcm_correlation(green)
    output[...,2:3] = fast_glcm_homogeneity(green)
    output[...,3:4] = fast_glcm_contrast(green)

    print('Calculating red band texture properties...')
    #output[...,4:5] = fast_glcm_correlation(red)
    output[...,4:5] = fast_glcm_contrast(red)

    print('Calculating nir band texture properties...')
    output[...,5:6] = fast_glcm_dissimilarity(nir)
    output[...,6:7] = fast_glcm_correlation(nir)
    output[...,7:8] = fast_glcm_contrast(nir)
    # output[...,8:9] = fast_glcm_homogeneity(nir)

    return output.astype(np.float32)
