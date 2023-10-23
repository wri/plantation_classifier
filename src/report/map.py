from typing import Text
import yaml
import boto3
import os
import pandas as pd
from utils.logs import get_logger


def tile_id_download(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    #   TODO: add check for connection errors
    #   TODO: add logic to compare local filedates for remote filedates
    tile_id_loc = config["evaluate"]["tile_id_loc"]
    dest_file = f"{tile_id_loc}/{location[1]}.csv"
    s3_file = f"2020/databases/{location[1]}.csv"
    logger = get_logger("TILE_ID", log_level=config["base"]["log_level"])
    # check if csv exists locally
    # confirm subdirectory exists otherwise download can fail
    if os.path.exists(dest_file):
        logger.info(f"Csv file for {location[1]} exists locally.")

        # if csv doesnt exist locally, check if available on s3
    if not os.path.exists(dest_file):
        boto3.setup_default_session(
            profile_name=config["data_load"]["sso_profile_name"]
        )
        conn = boto3.client("s3")
        bucket_name = config["evaluate"]["tile_s3_bucket"]

        # turn the bucket + file into a object summary list
        file_list = []
        for object in conn.list_objects_v2(Bucket=bucket_name, Prefix=prefix_string)[
            "Contents"
        ]:
            file_list.append(object["Key"])
        logger.info(s3_file, dest_file)

        if len(file_list) > 0:
            logger.info(f"The s3 resource s3://{bucket_name}/{s3_file} exists.")
            conn.download_file(Bucket=bucket_name, Prefix=s3_file, Filename=dest_file)

    database = pd.read_csv(dest_file)

    # create a list of tiles
    tiles = database[["X_tile", "Y_tile"]].to_records(index=False)

    return tiles

def download_ard(tile_idx: tuple, country: str, aws_access_key: str, aws_secret_key: str):
    '''
    If ARD folder is not present locally,
    Download contents from s3 folder into local folder
    for specified tile
    '''

    # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]

    s3_path = f'2020/ard/{str(x)}/{str(y)}/'
    s3_path_feats = f'2020/raw/{str(x)}/{str(y)}/raw/feats/'
    local_path = f'tmp/{country}/{str(x)}/{str(y)}/'

    # check if ARD folder has been downloaded
    ard_check = os.path.exists(local_path + 'ard/')
    feats_check = os.path.exists(local_path + 'raw/feats/')

    if not ard_check:
        print(f"Downloading ARD for {(x, y)}")

        s3 = boto3.resource('s3',
                            aws_access_key_id=aws_access_key,
                            aws_secret_access_key=aws_secret_key)
        bucket = s3.Bucket('tof-output')

        # this will download whatever is in the ard folder
        for obj in bucket.objects.filter(Prefix=s3_path):

            ard_target = os.path.join(local_path + 'ard/', os.path.relpath(obj.key, s3_path))
            print(f'target download path: {ard_target}')

            if not os.path.exists(os.path.dirname(ard_target)):
                os.makedirs(os.path.dirname(ard_target))
            if obj.key[-1] == '/':
                continue
            try:
                bucket.download_file(obj.key, ard_target)

            # if the tiles do not exist on s3, catch the error and return False
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False

    if not feats_check:
        print(f"Downloading feats for {(x, y)}")
        s3 = boto3.resource('s3',
                            aws_access_key_id=aws_access_key,
                            aws_secret_access_key=aws_secret_key)
        bucket = s3.Bucket('tof-output')

        for obj in bucket.objects.filter(Prefix=s3_path_feats):

            feats_target = os.path.join(local_path + 'raw/feats/', os.path.relpath(obj.key, s3_path_feats))
            print(f'target download path: {feats_target}')

            if not os.path.exists(os.path.dirname(feats_target)):
                os.makedirs(os.path.dirname(feats_target))
            if obj.key[-1] == '/':
                continue
            try:
                bucket.download_file(obj.key, feats_target)

            # if the tiles do not exist on s3, catch the error and return False
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False

    ard = hkl.load(f'{local_path}ard/{str(x)}X{str(y)}_ard.hkl')
    return ard, True

