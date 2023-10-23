import argparse
from typing import Text
import json
import pickle
import yaml
from utils.logs import get_logger
import features.prepare_data as prep


def featureize(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    logger = get_logger("FEATURIZE", log_level=config["base"]["log_level"])
    with open(config["data_load"]["ceo_json"], "r") as fp:
        ceo_batch_list = json.load(fp)
        logger.info("CEO survey targets loaded")
    X, y = prep.create_xy(
        ceo_batch_list, classes="multi", drop_feats=False, config_path=config_path
    )
    logger.info("X,y features loaded")
    X_train, X_test, y_train, y_test = prep.reshape_training_data(
        X, y, config_path, config["data_condition"]["scale_features"]
    )
    scale_data = config["data_condition"]["scale_features"]
    logger.info(f"Train and test set generated, scaled data: {scale_data}")
    with open(config["data_condition"]["train_data_x"], "wb") as fp:
        pickle.dump(X_train, fp)
    with open(config["data_condition"]["train_data_y"], "wb") as fp:
        pickle.dump(y_train, fp)
    with open(config["data_condition"]["test_data_x"], "wb") as fp:
        pickle.dump(X_test, fp)
    with open(config["data_condition"]["test_data_y"], "wb") as fp:
        pickle.dump(y_test, fp)
    logger.info("Training and testing data exported")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    featureize(config_path=args.config)
