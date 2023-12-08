#! /usr/bin/env python3

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rs
from rasterio.merge import merge
from rasterio.mask import mask
import os
import sys
from datetime import datetime
import boto3
import glob
import gc

def mosaic_tif(location: list, model: str, compile_from: str):

    ''''
    Takes in a list of tiles and merges them to form a single tif.
    Alternatively... merges all tifs in a folder.
    '''
    
    if not os.path.exists(f'tmp/{location[0]}/preds/mosaic/'):
        os.makedirs(f'tmp/{location[0]}/preds/mosaic/')
        
    # use the list of tiles to create a list of filenames
    # this will need to be updated to take in a specific list of tiles to mosaic
    tifs_to_mosaic = []

    if compile_from == 'csv':
        database = pd.read_csv(f'data/{location[1]}.csv')
        tiles = database[['X_tile', 'Y_tile']].to_records(index=False)

        # specify here if there's a specific set of tiles to merge
        for tile_idx in tiles[403:533]:
            x = tile_idx[0]
            y = tile_idx[1]
            filename = f'{str(x)}X{str(y)}Y_preds.tif'
            tifs_to_mosaic.append(filename)


    # get a list of files to merge from preds dir
    # filename slice specific to puntarenas -- update
    # use this in the event some tiles need to be skipped
    if compile_from == 'dir':
        tifs = glob.glob(f'../tmp/{location[0]}/preds/*.tif')  
        for file in tifs:
            tifs_to_mosaic.append(file[21:])

    # now open each item in dataset reader mode (required to merge)
    reader_mode = []

    for file in tifs_to_mosaic:
        src = rs.open(f'tmp/{location[0]}/preds/{file}')
        reader_mode.append(src) 

    print(f'Merging {len(reader_mode)} tifs.')

    mosaic, out_transform = merge(reader_mode)

    # delete the old list
    del tiles
    del tifs_to_mosaic
    del reader_mode

    date = datetime.today().strftime('%Y-%m-%d')
    
    # outpath will be the new filename 
    outpath = f'tmp/{location[0]}/preds/mosaic/{location[1]}_{model}_{date}_mrgd.tif'
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

def clip_it(location: list, model: str, shapefile: str):
    ''''
    imports a mosaic tif and clips it to the extent of a 
    given shapefile
    '''
    date = datetime.today().strftime('%Y-%m-%d')
    merged = f'tmp/{location[0]}/preds/mosaic/{location[1]}_{model}_{date}_mrgd.tif'
    shapefile = gpd.read_file(f'data/{shapefile}')
    clipped = f'tmp/{location[0]}/preds/mosaic/{location[1]}_{model}_{date}.tif'

    with rs.open(merged) as src:
        shapefile = shapefile.to_crs(src.crs)
        out_image, out_transform = mask(src, shapefile.geometry, crop=True)
        out_meta = src.meta.copy() 

    out_meta.update({
        "driver":"Gtiff",
        "height":out_image.shape[1], # height starts with shape[1]
        "width":out_image.shape[2], # width starts with shape[2]
        "transform":out_transform
    })

    with rs.open(clipped,'w',**out_meta) as dst:
        dst.write(out_image)
    os.remove(merged)
    
    return None


def upload_mosaic(location: list, model: str, aws_access_key: str, aws_secret_key: str):
    '''
    Uploads the combined tif to an s3 bucket
    '''
    date = datetime.today().strftime('%Y-%m-%d')
    
    # outpath will be the new filename
    suffix = f'{location[0]}_{model}_{date}.tif'
    mosaic_filepath = f'tmp/{location[0]}/preds/mosaic/{suffix}'

    s3 = boto3.resource('s3',
                        aws_access_key_id=aws_access_key, 
                        aws_secret_access_key=aws_secret_key)
    
    print(f'Uploading {mosaic_filepath} to s3.')

    s3.meta.client.upload_file(mosaic_filepath, 
                              'restoration-monitoring', 
                              'plantation-mapping/data/samples/' + suffix)

    return None


if __name__ == '__main__':
   
    import argparse
    parser = argparse.ArgumentParser()
    print("Argument List:", str(sys.argv))

    parser.add_argument('--country', dest='country', type=str)
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--compile', dest='compile_from', type=str)

    args = parser.parse_args()
    
    mosaic_tif(args.country, args.model, args.compile_from)