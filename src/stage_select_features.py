import argparse
from typing import Text
import yaml
import model.feature_selection as fsl
from utils.logs import get_logger


def select_features(param_path: Text) -> None:
    # load parameters
    with open(param_path) as file:
        params = yaml.safe_load(file)

    logger = get_logger("SELECT_FEATURES", log_level=params["base"]["log_level"])


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    train_model(param_path=args.params)
