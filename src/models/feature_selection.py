
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import shap

from catboost import CatBoostRegressor, Pool



def build_model(df, target):
    """
    Given the dataframe and target, build and return the model
    """
    # Set up target
    y = df[target]
    # Set up train-test split
    X = df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1066)
    # Identify the categorical features
    categorical_features = X_train.select_dtypes(exclude=[np.number])
    # Create training and test pools for catboost
    train_pool = Pool(X_train, y_train, categorical_features)
    test_pool = Pool(X_test, y_test, categorical_features)
    # Fit the model and calculate RMSE
    model = CatBoostRegressor(iterations=500, max_depth=5, learning_rate=0.05, random_seed=1066, logging_level='Silent')
    model.fit(X_train, y_train, eval_set=test_pool, cat_features=categorical_features, use_best_model=True, early_stopping_rounds=10)
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
    This function uses the SHAP importance from a catboost model
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
    


