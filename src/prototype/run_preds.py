#!/usr/bin/env python

import yaml
import confuse
import hickle as hkl
import pickle
import pandas as pd
import numpy as np
import os
from natsort import natsorted
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import h5py
from catboost import CatBoostClassifier
import sys
from datetime import datetime


def fit_eval_regressor(X_train, X_test, y_train, y_test, model_name, v_train_data):
    '''
    Based on arguments provided, fits and evaluates a regression model
    saving the model to a pkl file and saving score in a csv.
    '''

    if model_name == 'rfr':
        model = RandomForestRegressor(random_state=22)  
        model.fit(X_train, y_train)

    # save trained model
    filename = f'../models/{model_name}_model_{v_train_data}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
       
    # get scores and probabilities
    cv = cross_val_score(model, X_train, y_train, cv=3).mean()
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

     # add new scores
    scores = {'model': f'{model_name}_model_{v_train_data}', 
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
    csv. Prints out scores and visualizations for immediate review
    '''
    
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
        model = CatBoostClassifier(verbose=0, random_state=22)
        model.fit(X_train, y_train)
    
    # save trained model
    filename = f'../models/{model_name}_model_{v_train_data}.pkl'
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
    scores = {'model': f'{model_name}_model_{v_train_data}', 
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
    with open('../models/mvp_scores.csv', 'a') as f:
        f.write('\n')
    eval_df.to_csv('../models/mvp_scores.csv', mode='a', index=False, header=False)
    
    # doesn't work
    #eval_df.to_csv('../models/mvp_scores.csv', mode='a', index=False, header=False)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, pred, labels=model.classes_)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot();
 
    # ROC AUC and Precision Recall Curves
    plt.figure(figsize=(17,6)) 
    
    plt.subplot(1,2,1)
    
    # calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probs_pos)

    # plot roc curve and no skill model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=model_name, color='green')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right');
    
    plt.subplot(1,2,2)
    
    # calculate precision-recall curve
    fpr, tpr, thresholds = precision_recall_curve(y_test, probs_pos)

    # plot roc curve and no skill model
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0,1], [no_skill, no_skill], linestyle='--', label='No Skill')

    plt.plot(fpr, tpr, marker='.', label=model_name, color='purple')
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left');
    
    return pred, probs

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


def roc_curve_comp(X_train, X_test, y_train, y_test, model_names, v_train_data):
    
    plt.figure(figsize=(17,6)) 
    
    # ROC curve
    for m in model_names:
        
        with open(f'../models/{m}_model_{v_train_data}.pkl', 'rb') as file:  
             model = pickle.load(file)

        plt.subplot(1,2,1)
        
        # calculate and plot ROC curve
        probs = model.predict_proba(X_test)
        probs_pos = probs[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, probs_pos)
        plt.plot(fpr, tpr, marker=',', label=m)
    
    # plot no skill and custom settings
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right');
    
    # AUC curve
    for m in model_names:
        
        with open(f'../models/{m}_model_{v_train_data}.pkl', 'rb') as file:  
             model = pickle.load(file)

        plt.subplot(1,2,2)

        # calculate and plot precision-recall curve
        probs = model.predict_proba(X_test)
        probs_pos = probs[:, 1]
        fpr, tpr, thresholds = precision_recall_curve(y_test, probs_pos)
        plt.plot(fpr, tpr, marker=',', label=m)
    
    # plot no skill and custom settings
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0,1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.4, 1.05])
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left');
    
    return None


def learning_curve_comp(model_names, v_train_data, X_train, y_train):

    plt.figure(figsize = (13,6))
    
    colors = ['royalblue',
              'maroon', 
              'magenta', 
              'gold', 
              'limegreen'] 

    for i, x in zip(model_names, colors):

        filename = f'../models/round_1/{i}_model_{v_train_data}.pkl'

        with open(filename, 'rb') as file:
            model = pickle.load(file)

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, 
                                                                              X_train, 
                                                                              y_train, 
                                                                              cv=5, 
                                                                              return_times=True)

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.grid()
        plt.plot(train_sizes, train_scores_mean, "x-", color=x, label=f"{i} Train")
        plt.plot(train_sizes, test_scores_mean, ".-", color=x, label=f"{i} Test")
    
    plt.xlim([1000, 32000])
    plt.ylim([0.0, 1.2])
    plt.title('Comparison of Learning Curves')
    plt.xlabel('Training Samples')
    plt.ylabel('Score')
    plt.legend(loc='lower right');        
        
    return None


## TBD
def visualize_plotpreds(model_name, v_train_data, X_test):
    
    filename = f'../models/{model_name}_model_{v_train_data}.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    preds = model.predict(X_test)
 
    sns.heatmap(preds.reshape((14,14)), vmin=0, vmax=.8).set_title(model_name)

    return None