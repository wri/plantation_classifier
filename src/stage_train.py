import argparse
from typing import Text
import yaml
import json
import pickle
from utils.logs import get_logger
import models.feature_selection as fsl
import models.train as trn

def train(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    logger = get_logger('TRAIN', log_level=config['base']['log_level'])
    with open(config['data_condition']['train_data_x'], 'rb') as fp:
        X_train = pickle.load(fp)
    with open(config['data_condition']['train_data_y'], 'rb') as fp:
        y_train = pickle.load(fp)
    with open(config['data_condition']['test_data_x'], 'rb') as fp:
        X_test = pickle.load(fp)
    with open(config['data_condition']['test_data_y'], 'rb') as fp:
        y_test = pickle.load(fp)
    with open(config['data_condition']['val_data_x'], 'rb') as fp:
        X_val = pickle.load(fp)
    with open(config['data_condition']['val_data_y'], 'rb') as fp:
        y_val = pickle.load(fp)
    logger.info("Training, testing, and validation data loaded")
    
    if config['train']['select_features']:
        logger.info("Starting feature selection")
          
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    train(config_path=args.config)