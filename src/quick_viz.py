#!/usr/bin/env python

import seaborn as sns
import matplotlib.pyplot as plt
import rasterio as rs
import hickle as hkl
import numpy as np

# Base
# └── basic visualizations
#     ├── heatmaps
#     │   ├── multiply.py
#     │   ├── divide.py
#     ├── histograms
#     ├── confusion matrix

def heat_multiplot(matrices, cbarmin, cbarmax,  nrows = 13, ncols = 6):
    '''
    Type: Seaborn heatmap
    Purpose: Create a multiplot of heatmaps from a collection of matrices.

    Parameters:
    - matrices (array-like): A collection of matrices to be visualized as heatmaps.
    - cbarmin (float): The minimum value for the color bar scale.
    - cbarmax (float): The maximum value for the color bar scale.
    - nrows (int, optional): Number of rows for the subplot grid. Default is 13.
    - ncols (int, optional): Number of columns for the subplot grid. Default is 6.

    This function creates a multiplot of heatmaps from a collection of matrices. This function is most
    helpful if you want to visualize 3D arrays. It arranges the heatmaps in a grid with the specified 
    number of rows and columns. The color scale for the heatmaps is defined by the `cbarmin` 
    and `cbarmax` parameters. Each heatmap is displayed within its own subplot.

    Returns:
    None
    
    '''
    fig, axs = plt.subplots(ncols = ncols, nrows = nrows)
    fig.set_size_inches(18, 3.25*nrows)
    
    # create a list of indices from 0 to nrows*ncols
    to_iter = [[x for x in range(i, i + ncols + 1)] for i in range(0, nrows*ncols, ncols)]
    counter = 0
    
    #
    for r in range(1, nrows + 1):
        min_i = min(to_iter[r-1])
        max_i = max(to_iter[r-1])
        
        for i in range(ncols):
            sns.heatmap(data = matrices[..., counter], 
                        ax = axs[r - 1, i], 
                        cbar = True, 
                        vmin = cbarmin, # this could also be min_i
                        vmax = cbarmax, # this could also be max_i
                        cmap = sns.color_palette("viridis", as_cmap=True))
            axs[r - 1, i].set_xlabel("")
            axs[r - 1, i].set_ylabel("")
            axs[r - 1, i].set_yticks([])
            axs[r - 1, i].set_xticks([])
            counter += 1
        
    plt.show
    return None

def heat_compare_preds(location: str, tile_idx_a: tuple, tile_idx_b: tuple):
    '''
    Type: Seaborn heatmap
    Purpose: Compare and visualize multi-class predictions for two specific tiles in a given location 
    using heatmaps.

    Parameters:
    - location (str): The country.
    - tile_idx_a (tuple): The coordinates (x, y) of the first tile to compare.
    - tile_idx_b (tuple): The coordinates (x, y) of the second tile to compare.
    - title (str): The title to be displayed for the comparison plot.

    This function loads prediction data for two tiles specified by their coordinates (tile_idx_a and tile_idx_b)
    from the specified location directory and creates side-by-side heatmaps to visualize the predictions. The
    heatmaps display values in the range [0, 1, 2], and each tile's heatmap is shown in a separate subplot.

    Returns:
    None
    
    '''
    x_a, y_a = tile_idx_a[0], tile_idx_a[1]
    x_b, y_b = tile_idx_b[0], tile_idx_b[1]
    preds_a = rs.open(f'../tmp/{location}/preds/{str(x_a)}X{str(y_a)}Y_preds.tif').read(1)
    preds_b = rs.open(f'../tmp/{location}/preds/{str(x_b)}X{str(y_b)}Y_preds.tif').read(1)
    

    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    sns.heatmap(preds_a, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [0, 1, 2]}, 
                vmin=0, vmax=2).set_title('Tile: ' + str(tile_idx_a))
    plt.subplot(1,2,2)
    sns.heatmap(preds_b, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [0, 1, 2]}, 
                vmin=0, vmax=2).set_title('Tile: ' + str(tile_idx_b));

    return None

def hist_compare_s2(location: str, tile_idx_a: tuple, tile_idx_b: tuple, title:str, tile_idx_c: tuple = None):
    '''
    Type: Matplotlib histogram
    Purpose: Compare and visualize histograms of Sentinel-2 data for two specific tiles in a given location.

    Parameters:
    - location (str): The country.
    - tile_idx_a (tuple): The coordinates (x, y) of the first tile to compare.
    - tile_idx_b (tuple): The coordinates (x, y) of the second tile to compare.
    - title (str): The title for the histogram comparison plot, describing the area.

    This function loads analysis ready data for two tiles specified by their coordinates (tile_idx_a and tile_idx_b). 
    It then indexes the array to creates histograms of Sentinel-2 data and displays 
    them in a single plot for visual comparison.

    Returns:
    None
    
    '''
    x_a, y_a = tile_idx_a[0], tile_idx_a[1]
    x_b, y_b = tile_idx_b[0], tile_idx_b[1]
    ard_a = hkl.load(f'../tmp/{location}/{str(x_a)}/{str(y_a)}/ard/{str(x_a)}X{str(y_a)}Y_ard.hkl')
    ard_b = hkl.load(f'../tmp/{location}/{str(x_b)}/{str(y_b)}/ard/{str(x_b)}X{str(y_b)}Y_ard.hkl')
    
    s2_a = ard_a[..., 0:10]
    s2_b = ard_b[..., 0:10]
    
    plt.figure(figsize=(6,4))
    binwidth = .01
    min = s2_a.min()
    max = s2_a.max()

    # this asks for 33 binds between .01 and .6 -- np.arange(min, max + binwidth, binwidth)
    plt.hist(s2_a.flatten(), alpha=0.5, label=str(tile_idx_a), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
    plt.hist(s2_b.flatten(), alpha=0.3, label=str(tile_idx_b), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
    plt.xlim(0.0, 0.6)
    plt.xticks(np.arange(0.0, 0.6, 0.1))
    plt.title(title)
    plt.legend();

    if tile_idx_c is not None:
        x_c, y_c = tile_idx_c[0], tile_idx_c[1]
        ard_c = hkl.load(f'../tmp/{location}/{str(x_c)}/{str(y_c)}/ard/{str(x_c)}X{str(y_c)}Y_ard.hkl')
        s2_c = ard_c[..., 0:10]
        plt.hist(s2_c.flatten(), alpha=0.3, label=str(tile_idx_c), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
        plt.legend();

    return None

def hist_compare_s2_byband(location: str, tile_idx_a: tuple, tile_idx_b: tuple, title:str, tile_idx_c: tuple = None):
    '''
    Each s2 band is plot 
    
    '''
    x_a, y_a = tile_idx_a[0], tile_idx_a[1]
    x_b, y_b = tile_idx_b[0], tile_idx_b[1]
    ard_a = hkl.load(f'../tmp/{location}/{str(x_a)}/{str(y_a)}/ard/{str(x_a)}X{str(y_a)}Y_ard.hkl')
    ard_b = hkl.load(f'../tmp/{location}/{str(x_b)}/{str(y_b)}/ard/{str(x_b)}X{str(y_b)}Y_ard.hkl')
    
    s2_a = ard_a[..., 0:10]
    s2_b = ard_b[..., 0:10]

    plt.figure(figsize=(20,20))
    binwidth = .01
    min = s2_a.min()
    max = s2_a.max()
    band_counter = 0
    
    for i in range(1, 11):
        plt.subplot(4,3,i)
        plt.hist(s2_a[..., band_counter].flatten(), alpha=0.5, label=str(tile_idx_a), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
        plt.hist(s2_b[..., band_counter].flatten(), alpha=0.3, label=str(tile_idx_b), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
        plt.xlim(0.0, 0.5)
        plt.xticks(np.arange(0.0, 0.5, 0.1))
        plt.title(title + f' Band {str(band_counter)}')
        #plt.legend();

        if tile_idx_c is not None:
            x_c, y_c = tile_idx_c[0], tile_idx_c[1]
            ard_c = hkl.load(f'../tmp/{location}/{str(x_c)}/{str(y_c)}/ard/{str(x_c)}X{str(y_c)}Y_ard.hkl')
            s2_c = ard_c[..., 0:10]
            plt.hist(s2_c[..., band_counter].flatten(), alpha=0.3, label=str(tile_idx_c), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
        
        band_counter += 1

    return None




def heat_compare_hkl(arr_a, arr_b, title_a:str, title_b:str):
    '''
    Type: Seaborn heatmap
    Purpose: Compare and visualize two hickle files (could be s2 data, ARD, feats, etc.)
    
    '''

    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    sns.heatmap(arr_a, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [arr_a.min(), arr_a.max()]}).set_title(title_a)
        
    plt.subplot(1,2,2)
    sns.heatmap(arr_b, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [arr_b.min(), arr_b.max()]}).set_title(title_b);

    return None