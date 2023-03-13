#!/usr/bin/env python

import hickle as hkl
import pickle
import pandas as pd
import numpy as np
import os
from natsort import natsorted
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import h5py
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight

import sys
from datetime import datetime
import rasterio as rs
import validate_io as validate


def fit_eval_regressor(X_train, X_test, y_train, y_test, model_name, v_train_data):
    '''
    Based on arguments provided, fits and evaluates a regression model
    saving the model to a pkl file and saving score in a csv.
    '''
    # TODO: For multiregression models, loss_function should be 'MultiRMSE'

    # Get the count of features used
    tml_feat_count = X_train.shape[1] - 13
    
    if model_name == 'rfr':
        model = RandomForestRegressor(random_state=22)  
        model.fit(X_train, y_train)

    # save trained model
    filename = f'../models/{model_name}_{v_train_data}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
       
    # get scores and probabilities
    cv = cross_val_score(model, X_train, y_train, cv=3).mean()
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

     # add new scores
    scores = {'model': f'{model_name}_{v_train_data}', 
            'class': 'n/a',
            'tml_feats':tml_feat_count,
            'cv': cv, 
            'train_score': train_score, 
            'test_score': test_score, 
            'roc_auc': np.NaN,
            'precision': np.NaN,
            'recall': np.NaN,
            'f1': np.NaN,
            'date': datetime.today().strftime('%Y-%m-%d')}

    eval_df = pd.DataFrame([scores]).round(4)
        
    # write scores to new line of csv
    with open('../models/mvp_scores.csv', 'a') as f:
        f.write('\n')
        eval_df.to_csv('../models/mvp_scores.csv', mode='a', index=False, header=False)

    return eval_df


def fit_eval_classifier(X_train, X_test, y_train, y_test, model_name, v_train_data):
    
    '''
    Based on arguments provided, fits and evaluates a classification model
    saving the model to a pkl file and saving scores in a 
    csv. 
    '''
    # using this to check data scaling
    validate.model_inputs(X_train)

    # Get the count of features used
    tml_feat_count = X_train.shape[1] - 13

    # fit the selected classifier
    if model_name == 'rfc':
        model = RandomForestClassifier(random_state=22)  
        model.fit(X_train, y_train)
    
    elif model_name == 'lgbm':
        model = LGBMClassifier(random_state=22)
        model.fit(X_train, y_train)
        
    elif model_name == 'svm':
        model = SVC(probability=True, random_state=22)
        model.fit(X_train, y_train)
    
    elif model_name == 'xgb':
        model = XGBClassifier(use_label_encoder=False, random_state=22)
        model.fit(X_train, y_train)
    
    elif model_name == 'cat':
        # import param dist here
        # scale_pos_weight=0.381
        # depth=10, l2_leaf_reg=11, iterations=1100, learning_rate=0.02
        model = CatBoostClassifier(verbose=0, random_state=22)
        model.fit(X_train, y_train)
    
    # save trained model
    filename = f'../models/{model_name}_{v_train_data}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
       
    # get scores and probabilities
    cv = cross_val_score(model, X_train, y_train, cv=3).mean()
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    probs = model.predict_proba(X_test)
    pred = model.predict(X_test)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)    

    # calculate AUC score
    probs_pos = probs[:, 1]
    roc_auc = roc_auc_score(y_test, probs_pos)

    # add new scores to df
    scores = {'model': f'{model_name}_{v_train_data}', 
            'class': 'binary',
            'tml_feats':tml_feat_count,
            'cv': cv, 
            'train_score': train_score, 
            'test_score': test_score, 
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'date': datetime.today().strftime('%Y-%m-%d')}

    eval_df = pd.DataFrame([scores]).round(4)
    
    # write scores to new line of csv
    with open('../models/mvp_scores.csv', 'a', newline='') as f:
        f.write('\n')
    eval_df.to_csv('../models/mvp_scores.csv', mode='a', index=False, header=False)
    
    # doesn't work
    #eval_df.to_csv('../models/mvp_scores.csv', mode='a', index=False, header=False)
    
    return y_test, pred, probs, probs_pos



def fit_eval_multiclassifier(X_train, X_test, y_train, y_test, model_name, v_train_data):
    
    '''
    Fits and evaluates a CatBoost multi-classification (3 class) model
    saving the model to a pkl file and saving scores in a csv. 
    '''

    # estimates the class weights for unbalanced datsets
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # look into this optional parameter to disable automatic
    # metrics calculation (hints=skip_train~false)
    model = CatBoostClassifier(verbose=0, 
                              loss_function='MultiClass',
                              class_weights=class_weights,
                              random_state=22)
    
    model.fit(X_train, y_train)
    
    # save trained model
    filename = f'../models/{model_name}_{v_train_data}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
       
    # get scores and probabilities
    cv = cross_val_score(model, X_train, y_train, cv=3).mean()
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    probs = model.predict_proba(X_test)
    pred = model.predict(X_test)
    f1 = f1_score(y_test, pred, average='weighted')
    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')    

    # calculate AUC score
    # not sure about proper params 
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    probs_pos = probs[:, 1]
    #roc_auc = roc_auc_score(y_test, probs_pos, average='macro', multi_class='ovo')

    # add new scores to df
    scores = {'model': f'{model_name}_{v_train_data}', 
            'class': 'multi',
            'cv': cv, 
            'train_score': train_score, 
            'test_score': test_score, 
            'roc_auc': np.NaN,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'date': datetime.today().strftime('%Y-%m-%d')}

    eval_df = pd.DataFrame([scores]).round(4)
    
    # write scores to new line of csv
    with open('../models/mvp_scores.csv', 'a') as f:
        f.write('\n')
    eval_df.to_csv('../models/mvp_scores.csv', mode='a', index=False, header=False)

    
    return y_test, pred, probs, probs_pos


## TBD
def write_train_test_tif(arr: np.ndarray, bbx: list, tile_idx: tuple, country: str, suffix = "preds") -> str:
    '''
    Write predictions to a geotiff, using the same bounding box 
    to determine north, south, east, west corners of the tile
    '''
    
     # set x/y to the tile IDs
    x = tile_idx[0]
    y = tile_idx[1]
    out_folder = f'tmp/{country}/preds/'
    file = out_folder + f"{str(x)}X{str(y)}Y_{suffix}.tif"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # uses bbx to figure out the corners
    west, east = bbx[0], bbx[2]
    north, south = bbx[3], bbx[1]
    
    arr = arr.astype(np.uint8)

    # create the file based on the size of the array
    transform = rs.transform.from_bounds(west = west, south = south,
                                         east = east, north = north,
                                         width = arr.shape[1],
                                         height = arr.shape[0])

    print("Writing", file)
    new_dataset = rs.open(file, 'w', 
                            driver = 'GTiff',
                            height = arr.shape[0], width = arr.shape[1], count = 1,
                            dtype = "uint8",
                            compress = 'lzw',
                            crs = '+proj=longlat +datum=WGS84 +no_defs',
                            transform=transform)
    new_dataset.write(arr, 1)
    new_dataset.close()
    
    return None


