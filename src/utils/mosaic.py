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
import hickle as hkl
import boto3
import botocore

def download_shape(data_dir: str, 
                   location: list, 
                   bucket,
                   aws_access_key: str, 
                   aws_secret_key: str):
    '''
    Checks to see if the location shapefile exists locally,
    if not downloads the file from s3
    '''

    s3_dir = f'2020/shapefiles/{location[1]}/'
    dest_dir = f"{data_dir}shapefiles/"
    s3 = boto3.resource('s3',
                aws_access_key_id=aws_access_key, 
                aws_secret_access_key=aws_secret_key)
    bucket = s3.Bucket(bucket)
    
    # skip if shapefile present
    if os.path.exists(f"{data_dir}shapefiles/{location[1]}.shp"):
        print(f'Shapefile for {location[1]} exists locally.')

    # TODO: this should check if the file is present on s3 first
    else:
        for obj in bucket.objects.filter(Prefix=s3_dir):
            target = os.path.join(dest_dir, os.path.relpath(obj.key, s3_dir))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == '/':
                continue
            try:
                bucket.download_file(obj.key, target)

            # if the tiles do not exist on s3, catch the error and return False
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("Shapefile is not on s3.")
                    return False
                
        print(f"Shapefile downloaded for {location[1]}.")

    return True

def mosaic_tif(location: list, version: str, tiles: list):

    ''''
    Takes in a list of tiles from a csv file and 
    merges them to form a single tif.

    '''
    mosaic_dir = f'tmp/{location[0]}/preds/mosaic/'
    if not os.path.exists(mosaic_dir):
        os.makedirs(mosaic_dir)
        
    tifs_to_mosaic = []
    for tile_idx in tiles:
        x = tile_idx[0]
        y = tile_idx[1]
        filename = f'{str(x)}X{str(y)}Y_preds.tif'
        tifs_to_mosaic.append(filename)

    # now open each item in dataset reader mode (required to merge)
    reader_mode = []

    for file in tifs_to_mosaic:
        src = rs.open(f'tmp/{location[0]}/preds/{file}')
        reader_mode.append(src) 

    print(f'Merging {len(reader_mode)} tifs.')

    mosaic, out_transform = merge(reader_mode)

    date = datetime.today().strftime('%Y-%m-%d')
    
    # outpath will be the new filename 
    outpath = f'{mosaic_dir}{location[1]}_{version}_{date}_mrgd.tif'
    out_meta = src.meta.copy()  
    out_meta.update({'driver': "GTiff",
                     'dtype': 'uint8',
                     'height': mosaic.shape[1],
                     'width': mosaic.shape[2],
                     'transform': out_transform,
                     'compress':'lzw',
                     'nodata': 255})

    with rs.open(outpath, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Ensure to close all files
    del tiles
    del tifs_to_mosaic
    for src in reader_mode:
        src.close()

    return None

def clip_it(aak: str, 
            ask: str, 
            location: list, 
            version: str,
            dir:str,
            bucket: str):
    ''''
    imports a mosaic tif and clips it to the extent of a 
    given shapefile
    '''
    successful = download_shape(dir, location, bucket, aak, ask)
    
    mosaic_dir = f'tmp/{location[0]}/preds/mosaic/'
    date = datetime.today().strftime('%Y-%m-%d')
    merged = f'{mosaic_dir}{location[1]}_{version}_{date}_mrgd.tif'
    clipped = f'{mosaic_dir}{location[1]}_{version}_{date}.tif'
    
    if successful:
        shapefile = f"data/shapefiles/{location[1]}.shp"
        shapefile = gpd.read_file(shapefile)
        with rs.open(merged) as src:
            shapefile = shapefile.to_crs(src.crs)
            out_image, out_transform = mask(src, 
                                            shapefile.geometry, 
                                            crop=True)
            out_meta = src.meta.copy() 

        out_meta.update({
            "driver":"Gtiff",
            "height":out_image.shape[1], # height starts with shape[1]
            "width":out_image.shape[2], # width starts with shape[2]
            "transform":out_transform,
            "nodata": 255,
        })

        with rs.open(clipped,'w',**out_meta) as dst:
            dst.write(out_image)
        os.remove(merged)
        return clipped
    
    else:
        print("Shapefile does not exist - skipping clip.")
        return merged


def upload_mosaic(aak: str, 
                  ask: str, 
                  filename: str):
    '''
    Uploads the combined tif to an s3 bucket
    '''
    s3 = boto3.resource('s3',
                        aws_access_key_id=aak, 
                        aws_secret_access_key=ask)
   
    s3_filename = os.path.basename(filename)
    
    print(f'Uploading {s3_filename} to s3.')

    s3.meta.client.upload_file(filename, 
                              'restoration-monitoring', 
                              'plantation-mapping/data/samples/' + s3_filename)

    return None

# if __name__ == '__main__':
   
#     import argparse
#     parser = argparse.ArgumentParser()
#     print("Argument List:", str(sys.argv))

#     parser.add_argument('--country', dest='country', type=str)
#     parser.add_argument('--model', dest='model', type=str)
#     parser.add_argument('--compile', dest='compile_from', type=str)

#     args = parser.parse_args()
    
#     mosaic_tif(args.country, args.model, args.compile_from)