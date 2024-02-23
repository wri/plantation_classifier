from typing import Text
import yaml
import os
import boto3
from utils.logs import get_logger


def data_download(param_path: Text) -> None:
    '''
    A paginator is an iterator that will automatically paginate results for you. 
    You can use a paginator to iterate over the results of an operation.

    TODO: add check for connection errors
    TODO: add logic to compare local filedates for remote filedates
    TODO: update data comparison
    TODO: specify which CEO surveys will be downloaded
    '''

    with open(param_path) as param_file:
        params = yaml.safe_load(param_file)
    
    with open(params['base']['config_path']) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("S3_DOWNLOAD", log_level=params["base"]["log_level"])
    aak = config['aws']['aws_access_key_id']
    ask = config['aws']['aws_secret_access_key']
    bucket_name = params["data_load"]["bucket_name"]
    folder_prefix = params["data_load"]["folder_prefix"]
    data_prefix_list = params["data_load"]["data_prefix_list"]
    local_prefix = f'../{params["data_load"]["local_prefix"]}'

    s3 = boto3.client("s3",
                     aws_access_key_id=aak, 
                     aws_secret_access_key=ask)
    for prefix in data_prefix_list:
        logger.info("Downloading data for feature: " + prefix)
        prefix_string = f"{folder_prefix}train-{prefix}/"
        outpath = f"{local_prefix}/train-{prefix}/"
        file_list = []
        paginator = s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix_string)
        
        # create list of files to download in each dir
        for page in page_iterator:
            obj_dict = page["Contents"]
            for obj in obj_dict:
                # for object in conn.list_objects_v2(Bucket=bucket_name, Prefix=prefix_string)["Contents"]:
                file_list.append(obj["Key"])
        
        for file in file_list:
            s3.download_file(
                Bucket=bucket_name,
                Key=file,
                Filename=(outpath + os.path.basename(file)),
            )
    logger.info("Data download complete")