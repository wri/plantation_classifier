# copies training data to specified location from s3 if using SSO autentication 
# see SSO configuration setup at:
# https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html#sso-configure-profile-token-auto-sso
from typing import Text
import yaml
from utils.logs import get_logger

def data_download(config_path: Text) -> None:
    config_path='../params.yaml'
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    logger = get_logger('DATA_DOWNLOAD', log_level=config['base']['log_level'])
    if config['data_load']['download_data']:
#   TODO: add check for connection errors
#   TODO: add logic to compare local filedates for remote filedates
        boto3.setup_default_session(profile_name=config['data_load']['sso_profile_name'])
        bucket_name = config['data_load']['bucket_name']
        folder_prefix = config['data_load']['folder_prefix']
        data_prefix_list = config['data_load']['data_prefix_list']
        local_prefix = config['data_load']['local_prefix']
        conn = boto3.client('s3')
        for prefix in data_prefix_list:
            logger.info('Downloading data for feature: ' + prefix)
            prefix_string = f'{folder_prefix}train-{prefix}/'
            outpath = f'{local_prefix}/train-{prefix}/'
            file_list = []
            for object in conn.list_objects_v2(Bucket=bucket_name,Prefix=prefix_string)['Contents']:
                file_list.append(object['Key'])
    
            for file in file_list:
                conn.download_file(Bucket=bucket_name, Key=file, Filename=(outpath + os.path.basename(file)))
        logger.info('Data download complete')
    else: 
        logger.info('No new data downloaded')
        