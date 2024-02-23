import argparse
from typing import Text
import yaml
import json
from utils.logs import get_logger
from load_data import s3_download as download

def download(param_path: Text) -> None:
    '''
    These steps download data from s3 according to
    the specifed folders in the param file

    The tiles that will be processed are also identified
    TODO: update identify tiles to work for deply pipeline
    '''
    with open(param_path) as file:
        params = yaml.safe_load(file)

    # option to identify tiles and download by tile id?
    
    # download training data from s3
    logger = get_logger("LOAD_DATA", log_level=params["base"]["log_level"])
    if params["data_load"]["download_data"]:
        download.data_download(param_path) 
    else:
        logger.info("No new data downloaded.")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    download(param_path=args.params)


