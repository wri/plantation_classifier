import argparse
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from typing import Text, Dict
import yaml
import pickle

from utils.logs import get_logger
from reports.validation_visuals import plot_confusion_matrix

def convert_to_labels(indexes, labels):
    result = []
    for i in indexes:
        result.append(labels[i])
    return result


def write_confusion_matrix_data(y_true, predicted, labels, filename):
    '''

    '''
    assert len(predicted) == len(y_true)
    predicted_labels = convert_to_labels(predicted, labels)
    true_labels = convert_to_labels(y_true, labels)
    cf = pd.DataFrame(
        list(zip(true_labels, predicted_labels)), columns=["y_true", "predicted"]
    )
    cf.to_csv(filename, index=False)


def evaluate_model(params_path: Text) -> None:
    '''
    Evaluate model. Saves confusion matrix png and 
    data to reports folder.
    Args:
        params_path {Text}: path to params.yaml
    '''

    with open(params_path) as file:
        params = yaml.safe_load(file)

    logger = get_logger("EVALUATE", log_level=params["base"]["log_level"])

    logger.info("Loading model")
    model_dir = params["train"]["model_dir"]
    model_path = f"{model_dir}{params['train']['model_name']}.joblib"
    print(f'model path: {model_path}')
    model = joblib.load(f'{model_path}')

    # option to import selected features or full test data
    if params['train']['select_features']:
        with open(params["train"]["select_X_test"], "rb") as fp:
            X_test = pickle.load(fp)
    else:
        with open(params["data_condition"]["X_test"], "rb") as fp:
            X_test = pickle.load(fp)

    with open(params["data_condition"]["y_test"], "rb") as fp:
        y_test = pickle.load(fp)

    logger.info("Evaluating... (building report)")

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=prediction)
    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=prediction)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average="weighted")
    precision = precision_score(y_true=y_test, y_pred=prediction, average="weighted")
    recall = recall_score(y_true=y_test, y_pred=prediction, average="weighted")
    #logloss = log_loss(y_true=y_test, y_pred=prediction)

    cm = confusion_matrix(y_test, prediction)

    report = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # "logloss": logloss,
        "cm": cm,
        "actual": y_test,
        "predicted": prediction,
    }
   # logger.debug(report)
    # reports_dir = Path(params["evaluate"]["reports_dir"])
    metrics_path = params["evaluate"]["metrics_file"]
    logger.info(f'Writing metrics to {metrics_path}')
    with open(metrics_path, "w") as fp:
        json.dump(
            obj={
                "f1_score": f1,
                "accuracy_score": accuracy,
                "balanced_accuracy_socre": balanced_accuracy,
                "precision_score": precision,
                "recall_score": recall,
                # "logloss": logloss,
            },
            fp=fp,
        )

    logger.info("Creating confusion matrix")
    labels = list(set(y_test))
    labels = [int(fl) for fl in labels]

    # save confusion_matrix.png and data
    plt = plot_confusion_matrix(cm=report["cm"], target_names=labels, normalize=False)
    confusion_matrix_png_path = (params["evaluate"]["confusion_matrix_image"])
    plt.savefig(confusion_matrix_png_path)
    logger.info(f"Confusion matrix saved to : {confusion_matrix_png_path}")

    confusion_matrix_data_path = (params["evaluate"]["confusion_matrix_data"])
    y_test = [int(fl) for fl in y_test]
    prediction = [int(fl) for fl in prediction]
    write_confusion_matrix_data(y_test, 
                                prediction, 
                                labels=labels, 
                                filename=confusion_matrix_data_path
    )
    logger.info(f"Confusion matrix data saved to: {confusion_matrix_data_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    evaluate_model(params_path=args.params)