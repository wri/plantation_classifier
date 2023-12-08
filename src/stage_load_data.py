
##### PROTOTYPING ####

import argparse
from typing import Text
import yaml
import json
from utils.logs import get_logger
from load_data import identify_tiles
from load_data import s3_download as download

def identify_and_download(config_path: Text) -> None:
    '''
    These steps would access tiles to process from
    the CEO surveys and the code and cadence from 
    ptype_prepare_data.py
    '''
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    # could be an if statement here for loading train versus deployment tiles
    # identify CEO batches that will be used
    tiles = identify_tiles(config['data_load']['ceo_survey'])
    #logger.debug(ceo_batch_list)

    # what is the purpose of this list?
    with open(config["data_load"]["ceo_json"], "w") as fp:
        json.dump(obj={"CEO_survey_list": ceo_batch_list}, fp=fp)
    logger.info("CEO survey targets identified")

    # download data from s3
    logger = get_logger("DATA_LOAD", log_level=config["base"]["log_level"])
    if config["data_load"]["download_data"]:
        download.data_download(config_path, param_path) # TODO: clarify diff between param and configs
    else:
        logger.info("No new data downloaded.")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    identify_and_download(config_path=args.config)


