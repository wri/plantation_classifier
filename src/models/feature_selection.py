import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import shap



def build_model(X_train, X_test, y_train, y_test, estimator,
    model_params_dict, fit_params_dict):
    estimators = get_supported_estimator()
    if estimator_name not in estimators.keys():
        raise UnsupportedClassifier(estimator_name)
    estimator = estimators[estimator_name]()
    # Fit the model and calculate metric
    model = estimator(**model_params_dict)
    model.fit(X_train, y_train, **fit_params_dict)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    return rmse, model, X_test

def get_dropped_feature(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    feature_importance = shap_values.abs.mean(0).values
    importance_df = pd.DataFrame({'features': X_test.columns,
                                  'importance': feature_importance})
    importance_df.sort_values(by='importance', ascending=False, inplace=True)
    return importance_df['features'].iloc[-1]
    


def backward_selection(df, target, max_features=None):
    """
    This function uses the SHAP importance from a model
    to incrementally remove features from the training set until the RMSE no longer improves.
    This function returns the dataframe with the features that give the best RMSE.
    Return at most max_features.
    """
    # get baseline RMSE
    select_df = df.copy()
    total_features = df.shape[1]
    rmse, model, X_test = build_model(select_df, target)
    print(f"{rmse} with {select_df.shape[1]}")
    last_rmse = rmse
    
    # Drop least important feature and recalculate model peformance
    if max_features is None:
        max_features = total_features-1
        
    for num_features in range(total_features-1, 1, -1):
        # Trim features
        dropped_feature = get_dropped_feature(model, X_test)
        tmp_df = select_df.drop(columns=[dropped_feature])

        # Rerun modeling
        rmse, model, X_test = build_model(tmp_df, target)
        print(f"{rmse} with {tmp_df.shape[1]}")
        if (num_features < max_features) and (rmse > last_rmse):
            # RMSE increased, return last dataframe
            return select_df
        else:
            # RMSE improved, continue dropping features
            last_rmse = rmse
            select_df = tmp_df
    return select_df
    


