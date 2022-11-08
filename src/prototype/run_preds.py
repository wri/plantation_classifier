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
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import h5py
from catboost import CatBoostClassifier
import sys


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

    # add new scores
    scores = {'model': f'{model_name}_model_{v_train_data}', 
            'cv': cv, 
            'train_score': train_score, 
            'test_score': test_score, 
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1}

    eval_df = pd.DataFrame([scores]).round(4)
        
    # write scores to new line of csv
    # this is not working
    # with open('../models/mvp_scores.csv', 'a') as f:
    #     eval_df.to_csv('../models/mvp_scores.csv', mode='a', index=False, header=False)
    
    eval_df.to_csv('../models/mvp_scores.csv', mode='a', index=False, header=False)
    
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
    
    return eval_df


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

    plt.figure(figsize = (15,8))
    
    colors = ['royalblue',
              'maroon', 
              'magenta', 
              'gold', 
              'limegreen'] 

    for i, x in zip(model_names, colors):

        filename = f'../models/{i}_model_{v_train_data}.pkl'

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
        plt.plot(train_sizes, train_scores_mean, "x-", color=x, label=f"{i} Train score")
        plt.plot(train_sizes, test_scores_mean, "o-", color=x, label=f"{i} CV score")
    
    plt.xlim([1000, 32000])
    plt.ylim([0.0, 1.2])
    plt.title('Comparison of Learning Curves')
    plt.xlabel('Training Samples')
    plt.ylabel('Score')
    plt.legend(loc='lower right');        
        
    return None

