#! /usr/bin/env python3

import pandas as pd
import numpy as np
import rasterio as rs
from rasterio.merge import merge
import os
import sys


def mosaic_tif(country: str, model: str, date: str):

    ''''
    Takes in a list of tiles and merges them to form a single tif.
    '''

    # use the list of tiles to create a list of filenames
    # this will need to be updated to take in a specific list of tiles to mosaic
    tifs_to_mosaic = []

    database = pd.read_csv(f'data/{country}.csv')
    tiles = database[['X_tile', 'Y_tile']].to_records(index=False)

    # for now only mosaicing 20 tiles
    for tile_idx in tiles[:20]:
        x = tile_idx[0]
        y = tile_idx[1]
        filename = f'{str(x)}X{str(y)}Y_preds.tif'
        tifs_to_mosaic.append(filename)
    
    # tifs_to_mosaic = []
    # for tile in [x for x in os.listdir(f'../tmp/{country}/preds/') if x != 'mosaic' and x != '.DS_Store']:
    #     tifs_to_mosaic.append(tile)

    # potential to use this -- gdal not cooperating at the moment
    # gdal.BuildVRT(f'../tmp/preds/{filename}.vrt', tifs_to_mosaic, options=gdal.BuildVRTOptions(srcNodata=255, VRTNodata=255))
    # translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 255 -co BIGTIFF=YES"))
    # ds = gdal.Open(f'../tmp/preds/{filename}.vrt')
    # ds  = gdal.Translate(f'../tmp/preds/{filename}.tif', ds, options=translateoptions)
    
    # now open each item in dataset reader mode (required to merge)
    reader_mode = []
    for file in tifs_to_mosaic:
        src = rs.open(f'tmp/{country}/preds/{file}')
        reader_mode.append(src) 
    
    print(f'Merging {len(reader_mode)} tifs.')
    mosaic, out_transform = merge(reader_mode)
    
    # outpath will be the new filename
    suffix = f'{country}_{model}_{date}.tif'
    outpath = f'tmp/{country}/preds/mosaic/{suffix}'
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

    parser.add_argument('--country', dest='country', type=str)
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--date', dest='date', type=str)

    args = parser.parse_args()
    
    mosaic_tif(args.country, args.model, args.date)