#!/usr/bin/env python
import numpy as np
import hickle as hkl
import rasterio as rs
from rasterio.merge import merge
import copy
import pandas as pd
import sys

def make_bbox(country: str, tile_idx: tuple, expansion: int = 10) -> list:
    """
    Makes a (min_x, min_y, max_x, max_y) bounding box that
    is 2 * expansion 300 x 300 meter ESA LULC pixels. 

       Parameters:
            initial_bbx (list): [min_x, min_y, max_x, max_y]
            expansion (int): 1/2 number of 300m pixels to expand

       Returns:
            bbx (list): expanded [min_x, min_y, max_x, max_y]
    """
    bbx_df = pd.read_csv(f"data/{country}.csv", engine="pyarrow")

    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
    
    # extract the XY of interest as a dataframe
    bbx_df = bbx_df[bbx_df['X_tile'] == int(x)]
    bbx_df = bbx_df[bbx_df['Y_tile'] == int(y)]
    bbx_df = bbx_df.reset_index(drop = True)

    # creates a point [min x, min y, max x, max y] (min and max will be the same)
    initial_bbx = [bbx_df['X'][0], bbx_df['Y'][0], bbx_df['X'][0], bbx_df['Y'][0]]
    
    multiplier = 1/360 # Sentinel-2 pixel size in decimal degrees
    bbx = copy.deepcopy(initial_bbx)
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    
    # return the dataframe and the array
    return bbx

def preprocess_feats(tile_idx, country):

    x = tile_idx[0]
    y = tile_idx[1]
    folder = f'tmp/{country}/{str(x)}/{str(y)}/raw/feats/'
    tile_str = f'{str(x)}X{str(y)}Y'
    feats_raw = hkl.load(f'{folder}{tile_str}_feats.hkl').astype(np.float32)
    txt = np.load(f'{folder}{tile_str}_txt.npy')

    # adjust predictions feats[0] to match training data (0-1)
    # adjust shape by rolling axis 2x (65, 614, 618) ->  (614, 618, 65), (618, 614, 65) 
    # feats used for deply are multiplyed by 1000 before saving
    feats_raw[0, ...] = feats_raw[0, ...] / 100 
    feats_raw[1:, ...] = feats_raw[1:, ...] / 1000  
    feats_rolled = np.rollaxis(feats_raw, 0, 3)
    feats_rolled = np.rollaxis(feats_rolled, 0, 2)

    ttc = copy.deepcopy(feats_rolled) 
    high_feats = [np.arange(1,33)]
    low_feats = [np.arange(33,65)]
    ttc[:, :, [low_feats]] = feats_rolled[:, :, [high_feats]]
    ttc[:, :, [high_feats]] = feats_rolled[:, :, [low_feats]]

    n_feats = 65 + 16
    output = np.zeros((txt.shape[0], txt.shape[1], n_feats), dtype=np.float32)
    
    # combine ttc feats and txt into a single array
    output[..., :ttc.shape[-1]] = ttc
    output[..., ttc.shape[-1]:] = txt

    bbox = make_bbox(country, tile_idx)

    return output, bbox

def write_tif(arr: np.ndarray, bbx: list, tile_idx: tuple, country: str):

     # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
    out_folder = f'tmp/{country}/{str(x)}/{str(y)}/raw/feats/'
    file = out_folder + f"{str(x)}X{str(y)}Y_comb_feats.tif"

    # uses bbx to figure out the corners
    west, east = bbx[0], bbx[2]
    north, south = bbx[3], bbx[1]

    # create the file based on the size of the array (618, 614, 1)
    print("Writing", file)

    transform = rs.transform.from_bounds(west = west, south = south,
                                        east = east, north = north,
                                        width = arr.shape[1],  #614
                                        height = arr.shape[0]) #618
    new_dataset = rs.open(file, 'w', 
                            driver = 'GTiff',
                            width = arr.shape[1], 
                            height = arr.shape[0], 
                            count = 81,
                            dtype = "int16",
                            compress = 'lzw',
                            crs = '+proj=longlat +datum=WGS84 +no_defs',
                            transform=transform)
  
    new_dataset.write(arr)
    new_dataset.close()

    return None

def mosaic_neighbors(tile_a, tile_b, country):
    '''
    Takes two neighboring tiles and imports the raw ttc and txt 
    features. Performs preprocessing steps then
    merges the arrays into a single tif
    as output, with shape (614, 618, 81, 3)
    '''
    arr_a, bbox_a = preprocess_feats(tile_a, country)
    arr_b, bbox_b = preprocess_feats(tile_b, country)

    # convert arrays to int16 by rounding to 3 decimals and 
    # clipping to uint16 range for storage purposes
    arr_a = np.int16(np.clip(np.around(arr_a, 3), -3.2, 3.2) * 10000)
    arr_b = np.int16(np.clip(np.around(arr_b, 3), -3.2, 3.2) * 10000)

    write_tif(arr_a, bbox_a, tile_a, country)
    write_tif(arr_b, bbox_b, tile_b, country)

    tifs_to_mosaic = []
    for tile_idx in [tile_a, tile_b]:
        x = tile_idx[0]
        y = tile_idx[1]
        filename = f"tmp/{country}/{str(x)}/{str(y)}/raw/feats/{str(x)}X{str(y)}Y_comb_feats.tif"
        tifs_to_mosaic.append(filename)

    # now open each item in dataset reader mode (required to merge)
    reader_mode = []

    for file in tifs_to_mosaic:
        src = rs.open(file)
        reader_mode.append(src) 

    print(f'Merging {tile_a, tile_b}')

    mosaic, out_transform = merge(reader_mode)
    
    # outpath will be the new filename
    mosaic_dir = f'tmp/{country}/preds/mosaic/'
    outpath = f'{mosaic_dir}{tile_a}_{tile_b}.tif'
    out_meta = src.meta.copy()  
    out_meta.update({'driver': "GTiff",
                     'dtype': 'int16',
                     'height': mosaic.shape[1],
                     'width': mosaic.shape[2],
                     'transform': out_transform,
                     'compress':'lzw',
                     })

    with rs.open(outpath, "w", **out_meta) as dest:
        dest.write(mosaic)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', dest='country', type=str)
    parser.add_argument('--tile_a', dest='tile_a', nargs='+', type=int)
    parser.add_argument('--tile_b', dest='tile_b', nargs='+', type=int)
    args = parser.parse_args()
    print("Argument List:", args)
    tile_a = tuple(args.tile_a)
    tile_b = tuple(args.tile_b)
    mosaic_neighbors(tile_a, tile_b, args.country)