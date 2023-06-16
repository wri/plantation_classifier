import pandas as pd
import numpy as np
import hickle as hkl
import pickle
import seaborn as sns
import copy
import os
import boto3
import botocore
import rasterio as rs
import yaml
from osgeo import gdal
from scipy.ndimage import median_filter
from skimage.transform import resize
from glob import glob
import functools
from time import time, strftime
from datetime import datetime, timezone
from scipy import ndimage
from skimage.util import img_as_ubyte
import gc
from memory_profiler import profile
import copy
import subprocess

## import other scripts
import sys
sys.path.append('src/')

import mosaic
import validate_io as validate
import slow_glcm as slow_txt
import fast_glcm as fast_txt

with open("config.yaml", 'r') as stream:
    document = (yaml.safe_load(stream))
    aak = document['aws']['aws_access_key_id']
    ask = document['aws']['aws_secret_access_key']


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

# PIPELINE
# download_tile_ids()
# download ARD if folder not present - is this being downloaded or created?
# create and process TTC features - this could be it's own
# perform GLCM
# make_sample() but incorporating
# reshape
# run preds
# post process
# write tif
# remove tile
# execute per tif

def execute_per_tile(tile_idx: tuple, location: list, model, verbose: bool, feature_select: list):
    
    print(f'Processing tile: {tile_idx}')
    local_dir = 'tmp/' + location[0]
    successful = download_ard()

    if successful:
        validate.output_dtype_and_dimensions(s1_proc, s2_proc, dem_proc)
        sample, sample_dims = make_sample(dem_proc, s1_proc, s2_proc, feats)
        sample_ss = reshape_no_scaling(sample, verbose)
        
        validate.model_inputs(sample_ss)
        preds = predict_classification(sample_ss, model, sample_dims)
        preds_final = post_process_tile(preds, feature_select, no_data_flag, no_tree_flag)

        #validate.classification_scores(preds)
        write_tif(preds_final, bbx, tile_idx, location[0], 'preds')
        #remove_folder(tile_idx, local_dir)

        # clean up memory
        del bbx, s2_proc, s1_proc, dem_proc, feats, no_data_flag, no_tree_flag, sample, sample_ss, preds, preds_final
    
    else:
        print(f'Raw data for {tile_idx} does not exist on s3.')

    return None