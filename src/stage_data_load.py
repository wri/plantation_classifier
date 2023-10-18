import argparse
import joblib
from typing import Text
import yaml
from utils.logs import get_logger
from data.s3_download_sso import data_download

def data_load(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    logger = get_logger('DATA_DOWNLOAD', log_level=config['base']['log_level'])
    if config['data_load']['download_data']:
        data_download(config_path)
    else: 
        logger.info('No new data downloaded')
        
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    data_load(config_path=args.config)