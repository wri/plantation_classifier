#! /usr/bin/env python3

import pandas as pd
import numpy as np
import rasterio as rs
from rasterio.merge import merge
import os
import sys
sys.path.append('../src/')


def mosaic_tif(country: str, model: str):

    ''''
    Takes in a list of tiles and merges them to form a single tif.
    '''

    # use the list of tiles to create a list of filenames
    tifs_to_mosaic = []
    for tile in os.listdir(f'../tmp/{country}/preds/'):
        tifs_to_mosaic.append(tile)

    # potential to use this -- gdal not cooperating at the moment
    # gdal.BuildVRT(f'../tmp/preds/{filename}.vrt', tifs_to_mosaic, options=gdal.BuildVRTOptions(srcNodata=255, VRTNodata=255))
    # translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 255 -co BIGTIFF=YES"))
    # ds = gdal.Open(f'../tmp/preds/{filename}.vrt')
    # ds  = gdal.Translate(f'../tmp/preds/{filename}.tif', ds, options=translateoptions)
    
    # now open each item in dataset reader mode (required to merge)
    reader_mode = []
    for file in tifs_to_mosaic:
        src = rs.open(f'../tmp/{country}/preds/{file}')
        reader_mode.append(src) 
    
    print(f'Merging {len(reader_mode)} tifs...')
    mosaic, out_transform = merge(reader_mode)
    
    # outpath will be the new filename
    outpath = f'../tmp/{country}/preds/{country}_{model}.tif'
    out_meta = src.meta.copy()
    out_meta.update({'driver': "GTiff",
                     'dtype': 'uint8',
                     'height': mosaic.shape[1],
                     'width': mosaic.shape[2],
                     'transform': out_transform,
                     'compress':'lzw'})

    with rs.open(outpath, "w", **out_meta) as dest:
        dest.write(mosaic)

    return None





if __name__ == '__main__':
   
    import argparse
    parser = argparse.ArgumentParser()
    print("Argument List:", str(sys.argv))

    parser.add_argument('--tile_list', dest='tiles', nargs='+', type=list)
    parser.add_argument('--country', dest='country', type=str)
    parser.add_argument('--model', dest='model', type=str)

    args = parser.parse_args()
    
    mosaic_tif(args.tiles, args.country, args.model)