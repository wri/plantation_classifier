import argparse
from typing import Text
import yaml
import json
from utils.logs import get_logger
from data.s3_download_sso import data_download
from data.clean_ceo_summary import import_ceo_summary, id_ceo_batches

def data_load(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    logger = get_logger('DATA_LOAD', log_level=config['base']['log_level'])
    if config['data_load']['download_data']:
        data_download(config_path)
    else: 
        logger.info('No new imagery data downloaded')
    ceo_summary = import_ceo_summary(config_path)
    ceo_batch_list = id_ceo_batches(config_path, ceo_summary)
    with open(config['data_load']['ceo_json'], "w") as fp:
        json.dump(ceo_batch_list, fp)
    logger.info('CEO survey targets identified')
    
        
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    data_load(config_path=args.config)