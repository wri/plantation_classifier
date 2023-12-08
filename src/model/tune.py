#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from datetime import datetime


def random_search_cat(X_train, y_train, model):
    '''
    Performs a randomized search of hyperparameters using Catboost's built in
    random search method and plots the results, then
    and saves results to a csv file

    iterations: specifies the number of boosting iterations (trees) used during training (equiv to n_estimators)
    learning_rate: controls step size at each iteration while moving toward a min of the loss function (decrease if overfitting)
    depth: Determines the max depth of the individual decision trees (equiv to max_depth (must be <= 16))
    l2_leaf_reg: Regularization term that prevents overfitting by penalizing large parameter values.
    loss_function: Specifies the loss function to be optimized during training. 
    For regression tasks, you might use RMSE, while for classification, Logloss is common.
    '''
    # Get the count of features used
    feat_count = X_train.shape[1] - 13
    
    iterations = [int(x) for x in np.linspace(500, 1200, 10)]            
    depth = [int(x) for x in np.linspace(4, 10, 4)]                  
    l2_leaf_reg = [int(x) for x in np.linspace(2, 30, 4)]
    learning_rate = [.02, .03, .04]                                      

    param_dist = {'iterations': iterations,
                  'depth': depth,
                  'l2_leaf_reg': l2_leaf_reg,
                  'learning_rate': learning_rate}

    # instantiate the classifier and perform Catboost built in method for random search
    cat = CatBoostClassifier(random_state=22, 
                             loss_function='MultiClass', 
                             verbose=False)
    
    randomized_search_result = cat.randomized_search(param_dist,
                                                     X=X_train,
                                                     y=y_train,
                                                     n_iter=30,
                                                     cv=3,
                                                     plot=True,
                                                     verbose=False)
        
    rs_results = {'model': {model},
                  'class': 'binary',
                  'tml_feats': feat_count,
                  'iterations': iterations,
                  'depth': depth,
                  'l2_leaf_reg': l2_leaf_reg,
                  'learning_rate': learning_rate,
                  'results': randomized_search_result,
                  'date': datetime.today().strftime('%Y-%m-%d')}

    df = pd.DataFrame([rs_results])
    print(df)
    
    # write scores to new line of csv
    with open('../models/random_search.csv', 'a', newline='') as f:
        f.write('\n')
        df.to_csv('../models/random_search.csv', mode='a', index=False, header=False)
    
    return randomized_search_result
