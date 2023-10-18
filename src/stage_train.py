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
        estimator_name = config['train']['estimator_name']
        select_X_train, select_X_test = fsl.backward_selection(X_train, X_test, 
                                                y_train, y_test, 
                                                config['train']['estimator_name'], 
                                                config['train']['tuning_metric'],
                                                config['train']['estimators'][estimator_name]['param_grid'], 
                                                config['train']['fit_params'], 
                                                logger, 
                                                config['train']['max_features'])
        logger.info(f"Feature selection completed with {select_X_train.shape[1]} features")
        X_train = select_X_train
        X_test = select_X_test
    else:
        logger.info("Using all features")
    if config['train']['tune_hyperparams']:
        logger.info("Starting hyperparameter tuning")
    else:
        logger.info("Using default hyperparameters")

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    train(config_path=args.config)