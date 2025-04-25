# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2
import functools
from datetime import datetime
 

### CODE ADAPTED FROM https://github.com/tzm030329/GLCM/blob/master/fast_glcm.py
### Credit: Taka Izumi

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

@timer
def fast_glcm(img, vmin, vmax, levels, kernel_size, distance, angle):
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
    shape = (levels, levels, h, w) or for deployment pipeline (255, 255, 618, 614)
    dtype = float32
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

    # create a mask for each grey level
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1
    
    # create kernal (window) to perform filtering on glcm
    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)
    
    glcm = glcm.astype(np.float32)

    return glcm


@timer
def fast_contrast(glcm, levels):
    '''
    calc glcm contrast
    '''
    cont = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            cont += glcm[i,j] * (i-j)**2

    return cont[:,:,np.newaxis]

@timer
def fast_dissimilarity(glcm, levels):
    '''
    calc glcm dissimilarity
    '''

    diss = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            diss += glcm[i,j] * np.abs(i-j)

    return diss[:,:,np.newaxis]

@timer
def fast_homogeneity(glcm, levels):
    '''
    calc glcm homogeneity
    '''

    homo = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            homo += glcm[i,j] / (1.+(i-j)**2)

    return homo[:,:,np.newaxis]

@timer
def fast_correlation(glcm, levels):

    # this takes skimage implementation
    # and applies with fast glcm infrastructure
    # glcm has shape (255, 255, 618, 614)
    
    P = glcm
    num_level = levels

    #normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    I, J = np.ogrid[0:num_level, 0:num_level]
    corr = np.zeros((P.shape[2], P.shape[3]), dtype=np.float32)

    I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
    J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
    diff_i = I - np.sum(I * P, axis=(0, 1))
    diff_j = J - np.sum(J * P, axis=(0, 1))

    std_i = np.sqrt(np.sum(P * (diff_i) ** 2, axis=(0, 1)))
    std_j = np.sqrt(np.sum(P * (diff_j) ** 2, axis=(0, 1)))
    cov = np.sum(P * (diff_i * diff_j), axis=(0, 1))

    # handle the special case of standard deviations near zero
    mask_0 = std_i < 1e-15
    mask_0[std_j < 1e-15] = True
    corr[mask_0] = 1

    # handle the standard case
    mask_1 = ~mask_0
    corr[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])

    return corr[:,:,np.newaxis]

@timer
def extract_texture(img, properties_list, pipeline):

    ''''
    documentation
    Bands must be provided as img in this order: blue, green, red, nir
    Properties_list must be provided in this order: ['dissimilarity', 'correlation', 'homogeneity', 'contrast'] 

    create option arg for train/deployment pipeline.
    '''

    vmin, vmax = 0, 255 
    levels = 255
    ks = 5
    distance = 1.0
    angle = 0.0

    if pipeline == 'train':
        h,w = (14, 14)
    else:
        h,w = img.shape

    texture_arr = np.zeros((h, w, len(properties_list)), dtype=np.float32)
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    index = 0

    for prop in properties_list:
         
        if prop == 'dissimilarity':
            output = fast_dissimilarity(glcm, levels)
        
        elif prop == 'correlation':
            output = fast_correlation(glcm, levels)

        elif prop == 'homogeneity':
            output = fast_homogeneity(glcm, levels)
        
        elif prop == 'contrast':
            output = fast_contrast(glcm, levels)

        if pipeline == 'train':
            border_x = (output.shape[0] - 14) // 2
            border_y = (output.shape[1] - 14) // 2
            output = output[border_x:-border_x, border_y:-border_y]

        # append property to output array (618, 614, 1)
        texture_arr[..., index:index+1] = output
        index += 1

    return texture_arr


def old_fast_glcm_train(img):
    '''
    Bands must be calculated in this order: blue, green, red, nir
    Texture must be calculatd in this order: dissimilarity, correlation, homogeneity, contrast 
    '''
    vmin, vmax = 0, 255
    levels=8
    ks=5
    distance=1.0
    angle=0.0

    blue = img[..., 0]
    green = img[..., 1]
    red = img[..., 2]
    nir = img[..., 3]

    output = np.zeros((14, 14, 16), dtype=np.float32)

    output[...,0:1] = fast_dissimilarity(blue, vmin, vmax, levels, ks, distance, angle)
    output[...,1:2] = fast_correlation(blue, vmin, vmax, levels, ks, distance, angle)
    output[...,2:3] = fast_homogeneity(blue, vmin, vmax, levels, ks, distance, angle)
    output[...,3:4] = fast_contrast(blue, vmin, vmax, levels, ks, distance, angle)

    output[...,4:5] = fast_dissimilarity(green, vmin, vmax, levels, ks, distance, angle)
    output[...,5:6] = fast_correlation(green, vmin, vmax, levels, ks, distance, angle)
    output[...,6:7] = fast_homogeneity(green, vmin, vmax, levels, ks, distance, angle)
    output[...,7:8] = fast_contrast(green, vmin, vmax, levels, ks, distance, angle)

    output[...,8:9] = fast_dissimilarity(red, vmin, vmax, levels, ks, distance, angle)
    output[...,9:10] = fast_correlation(red, vmin, vmax, levels, ks, distance, angle)
    output[...,10:11] = fast_homogeneity(red, vmin, vmax, levels, ks, distance, angle)
    output[...,11:12] = fast_contrast(red, vmin, vmax, levels, ks, distance, angle)

    output[...,12:13] = fast_dissimilarity(nir, vmin, vmax, levels, ks, distance, angle)
    output[...,13:14] = fast_correlation(nir, vmin, vmax, levels, ks, distance, angle)
    output[...,14:15] = fast_homogeneity(nir, vmin, vmax, levels, ks, distance, angle)
    output[...,15:16] = fast_contrast(nir, vmin, vmax, levels, ks, distance, angle)

    return output.astype(np.float32)




