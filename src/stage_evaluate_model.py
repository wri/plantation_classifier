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
)
from typing import Text, Dict
import yaml
import pickle

from utils.logs import get_logger
#from report.visualize import plot_confusion_matrix

def convert_to_labels(indexes, labels):
    result = []
    for i in indexes:
        result.append(labels[i])
    return result


def write_confusion_matrix_data(y_true, predicted, labels, filename):
    assert len(predicted) == len(y_true)
    predicted_labels = convert_to_labels(predicted, labels)
    true_labels = convert_to_labels(y_true, labels)
    cf = pd.DataFrame(
        list(zip(true_labels, predicted_labels)), columns=["y_true", "predicted"]
    )
    cf.to_csv(filename, index=False)


def evaluate_model(params_path: Text) -> None:
    '''
    Evaluate model.
    Args:
        params_path {Text}: path to params.yaml
    '''
    with open(params_path) as file:
        params = yaml.safe_load(file)

    logger = get_logger("EVALUATE", log_level=params["base"]["log_level"])

    logger.info("Load model")
    model_path = params["train"]["model_path"]
    model = joblib.load(model_path)

    logger.info("Load test dataset")
    with open(params["train"]["select_X_test"], "rb") as fp:
        X_test = pickle.load(fp)
    with open(params["data_condition"]["y_test"], "rb") as fp:
        y_test = pickle.load(fp)

    logger.info("Evaluate (build report)")

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=prediction)
    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=prediction)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average="weighted")
    precision = precision_score(y_true=y_test, y_pred=prediction, average="weighted")
    recall = recall_score(y_true=y_test, y_pred=prediction, average="weighted")

    cm = confusion_matrix(y_test, prediction)

    report = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
        "actual": y_test,
        "predicted": prediction,
    }
    logger.debug(report)
    logger.info("Save metrics")
    # save f1 metrics file
    reports_folder = Path(params["evaluate"]["reports_dir"])
    metrics_path = reports_folder / params["evaluate"]["metrics_file"]

    json.dump(
        obj={
            "f1_score": f1,
            "accuracy_score": accuracy,
            "balanced_accuracy_socre": balanced_accuracy,
            "precision_score": precision,
            "recall_score": recall,
        },
        fp=open(metrics_path, "w"),
    )

    logger.info(f"Metrics file saved to : {metrics_path}")

    logger.info("Save confusion matrix")
    labels = list(set(y_test))
    labels = [int(fl) for fl in labels]

    # # save confusion_matrix.png
    # plt = plot_confusion_matrix(cm=report["cm"], target_names=labels, normalize=False)
    # confusion_matrix_png_path = (
    #     reports_folder / params["evaluate"]["confusion_matrix_image"]
    # )
    # plt.savefig(confusion_matrix_png_path)
    # logger.info(f"Confusion matrix saved to : {confusion_matrix_png_path}")

    # confusion_matrix_data_path = (
    #     reports_folder / params["evaluate"]["confusion_matrix_data"]
    # )
    # y_test = [int(fl) for fl in y_test]
    # prediction = [int(fl) for fl in prediction]
    # # logger.info(y_test)
    # write_confusion_matrix_data(
    #     y_test, prediction, labels=labels, filename=confusion_matrix_data_path
    # )
    # logger.info(f"Confusion matrix data saved to : {confusion_matrix_data_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()

    evaluate_model(params_path=args.params)