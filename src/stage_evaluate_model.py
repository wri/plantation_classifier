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
from evaluation.validation_visuals import plot_confusion_matrix
from features import ModelData


def convert_to_labels(indexes, labels):
    result = []
    for i in indexes:
        result.append(labels[i])
    return result


def write_confusion_matrix_data(y_true, predicted, labels, filename):
    """ """
    assert len(predicted) == len(y_true)
    predicted_labels = convert_to_labels(predicted, labels)
    true_labels = convert_to_labels(y_true, labels)
    cf = pd.DataFrame(
        list(zip(true_labels, predicted_labels)), columns=["y_true", "predicted"]
    )
    cf.to_csv(filename, index=False)


def evaluate_model(params_path: Text) -> None:
    """
    Evaluate model. Saves confusion matrix png and
    data to reports folder.
    Args:
        params_path {Text}: path to params.yaml
    """

    with open(params_path) as file:
        params = yaml.safe_load(file)

    logger = get_logger("EVALUATE", log_level=params["base"]["log_level"])

    logger.info("Loading model and test data")
    #   pipe = params['base']['pipeline']
    model_path = f"{params['train']['model_name']}"
    model = joblib.load(f"{model_path}")

    with open(params["data_condition"]["modelData_path"], "rb") as fp:
        model_data = pickle.load(fp)
    with open(params["select"]["selected_features_path"], "r") as fp:
        selected_features = json.load(fp)

    model_data.filter_features(selected_features)
    #   model_params["class_weights"] = model_data.class_weights

    logger.info("Evaluating (building report)")
    #  y_test = y_test.astype("str")

    prediction = model.predict(model_data.X_test_reshaped)
    accuracy = accuracy_score(y_true=model_data.y_test_reshaped, y_pred=prediction)
    balanced_accuracy = balanced_accuracy_score(
        y_true=model_data.y_test_reshaped, y_pred=prediction
    )
    f1 = f1_score(
        y_true=model_data.y_test_reshaped, y_pred=prediction, average="weighted"
    )
    precision = precision_score(
        y_true=model_data.y_test_reshaped, y_pred=prediction, average="weighted"
    )
    recall = recall_score(
        y_true=model_data.y_test_reshaped, y_pred=prediction, average="weighted"
    )

    cm = confusion_matrix(model_data.y_test_reshaped, prediction)

    report = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
        "actual": model_data.y_test_reshaped,
        "predicted": prediction,
    }
    metrics_path = f'{params["evaluate"]["metrics_file"]}.json'
    logger.info(f"Writing metrics to {metrics_path}")
    with open(metrics_path, "w") as fp:
        json.dump(
            obj={
                "f1_score": f1,
                "accuracy_score": accuracy,
                "balanced_accuracy_socre": balanced_accuracy,
                "precision_score": precision,
                "recall_score": recall,
            },
            fp=fp,
        )

    # converts ['0.0', '2.0', '3.0', '1.0'] to [0, 2, 3, 1]
    logger.info("Creating confusion matrix")
    labels = list(set(model_data.y_test_reshaped))
    labels = [int(float(fl)) for fl in labels]

    # save confusion_matrix.png and data
    plt = plot_confusion_matrix(cm=report["cm"], target_names=labels, normalize=False)
    confusion_matrix_png_path = f'{params["evaluate"]["cm_image"]}.png'
    plt.savefig(confusion_matrix_png_path)
    logger.info(f"Confusion matrix saved to : {confusion_matrix_png_path}")

    confusion_matrix_data_path = f'{params["evaluate"]["cm_data"]}.csv'
    y_test = [int(float(fl)) for fl in model_data.y_test_reshaped]
    prediction = [int(float(fl)) for fl in prediction]
    write_confusion_matrix_data(
        y_test, prediction, labels=labels, filename=confusion_matrix_data_path
    )
    logger.info(f"Confusion matrix data saved to: {confusion_matrix_data_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    evaluate_model(params_path=args.params)
