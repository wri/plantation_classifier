import catboost as cb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import optuna
import models.train as trn
from utils.logs import get_logger


def objective(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    metric, model, X_test = trn.train(
        X_train,
        X_test,
        y_train,
        y_test,
        estimator_name,
        metric_name,
        model_params_dict,
        fit_params_dict,
    )

    model = cb.CatBoostClassifier(**params, silent=True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    metric = mean_squared_error(y_val, predictions, squared=False)
    return metric


op_study = optuna.create_study(direction="maximize")

op_study.optimize(objective, n_trials=30)
