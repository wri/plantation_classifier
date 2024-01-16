import argparse
from typing import Text
import yaml
import joblib
import pickle
import json
from pathlib import Path
from utils.logs import get_logger
import model.train as train
import model.feature_selection as fsl
import model.tune as tune


def train_model(param_path: Text) -> None:
    """ """

    # load training parameters
    with open(param_path) as file:
        params = yaml.safe_load(file)

    logger = get_logger("TRAIN", log_level=params["base"]["log_level"])
    perform_scaling = params["train"][
        "perform_fs"
    ]  # scaling is only done if fs performed
    perform_fs = params["train"]["perform_fs"]
    perform_hp = params["train"]["tune_hyperparams"]
    logger.info(f"Perform scaling: {perform_scaling}")
    logger.info(f"Perform feature selection: {perform_fs}")
    logger.info(f"Perform tuning: {perform_hp}")

    with open(params["data_condition"]["X_train"], "rb") as fp:
        X_train = pickle.load(fp)
    logger.debug(f"X_train shape: {X_train.shape}")
    with open(params["data_condition"]["y_train"], "rb") as fp:
        y_train = pickle.load(fp)
    logger.debug(f"y_train shape: {y_train.shape}")
    with open(params["data_condition"]["X_test"], "rb") as fp:
        X_test = pickle.load(fp)
    logger.debug(f"X_test shape: {X_test.shape}")
    with open(params["data_condition"]["y_test"], "rb") as fp:
        y_test = pickle.load(fp)
    logger.debug(f"y_test shape: {y_test.shape}")
    logger.info("Training and testing data loaded.")

    estimator_name = params["train"]["estimator_name"]
    model_dir = Path(params["train"]["model_dir"])
    model_path = model_dir / params["train"]["model_name"]

    if perform_hp:
        logger.info(f"Starting random search with {params['tune']['n_iter']} samples")
        tuning_params, tuned_model = tune.random_search_cat(
            X_train,
            X_test,
            y_train,
            y_test,
            params["train"]["estimator_name"],
            params["train"]["tuning_metric"],
            logger,
            param_path,
        )
        logger.info(f"Best hyperparameters: {tuning_params['params']}")
        logger.info(f"Saving best hyperparameters to {params["tune"]["best_params"]}")
        # save dvc version
        with open(params["tune"]["best_params"], "w") as fp:
            json.dump(obj=tuning_params, fp=fp)
        # save other untracked version - hard coded for now
        #        with open('models/model_specs/best_params.json', "w") as fp:
        #            json.dump(obj=tuning_params,
        #                    fp=fp,
        #                    )
        logger.info("Saving tuned model")
        joblib.dump(tuned_model, f"{model_path}.joblib")

    else:
        logger.info(f"Training model with given (or default) hyperparameters")
        metric, model, X_test = train.fit_estimator(
            X_train,
            X_test,
            y_train,
            y_test,
            estimator_name,
            params["train"]["tuning_metric"],
            params["train"]["estimators"][estimator_name]["param_grid"],
            logger,
        )
        logger.info("Saving model")
        joblib.dump(model, f"{model_path}.joblib")
        with open(params["tune"]["best_params"], "w") as fp:
            json.dump({}, fp=fp)

        if perform_fs:
            max_features = params["train"]["max_features"]
            assert max_features <= 81
            logger.info(f"Max features for feature selection: {max_features}")
            top_feats = fsl.feature_importance(model_path, logger, max_features)
            logger.info(f"Writing features to file")
            # save dvc version
            with open(params["data_condition"]["selected_features"], "w") as fp:
                json.dump(
                    obj={"feature_index": top_feats, "n_features": len(top_feats)},
                    fp=fp,
                )
            # save other untracked version - hard coded for now
            with open("models/model_specs/selected_features.json", "w") as fp:
                json.dump(
                    obj={"feature_index": top_feats, "n_features": len(top_feats)},
                    fp=fp,
                )

        else:
            with open(params["data_condition"]["selected_features"], "wb") as fp:
                pickle.dump({}, fp)

    # fs with backward selection and shap analysis
    # if perform_fs:
    #     max_features = params["train"]["max_features"]
    #     logger.info(f"Max features for feature selection: {max_features}")
    #     select_X_train, select_X_test, fs_model = fsl.backward_selection(
    #         X_train,
    #         X_test,
    #         y_train,
    #         y_test,
    #         params["train"]["estimator_name"],
    #         params["train"]["tuning_metric"],
    #         params["train"]["estimators"][estimator_name]["param_grid"],
    #         logger,
    #         max_features)
    #     logger.info(f"Feature selection completed with {select_X_train.shape[1]} features")

    # overwrite saved X_train and X_test w feature selected data
    # do we want to save this data?
    # logger.info("Saving feature selected model and data.")
    # joblib.dump(fs_model, f'{model_path}.joblib')
    # with open(params["data_condition"]["X_train"], "wb") as fp:
    #     pickle.dump(select_X_train, fp)
    # with open(params["data_condition"]["X_test"], "wb") as fp:
    #     pickle.dump(select_X_test, fp)

    # logger.info(f'Writing features to file')
    # with open(params["data_condition"]["selected_features"], "w") as fp:
    #     json.dump(
    #         obj={"feature_index": list(select_X_test.columns),
    #             "n_features": len(list(select_X_test.columns))},
    #         fp=fp,
    #         )

    # or train model with all feats and create empty file as placeholder
    # else:
    #     with open(params["data_condition"]["selected_features"], "wb") as fp:
    #         pickle.dump({}, fp)

    # if performing hyperparameter tuning
    # load data from scratch w/ feature selection rather than
    # using arrays from above


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    train_model(param_path=args.params)
