#!/usr/bin/env python

from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np


def cm_roc_pr(model, y_test, pred, probs_pos):

    ''' 
    Produces a confusion matrix, ROC curve and precision recall curve
    '''
    
    with open(f'../models/{model}.pkl', 'rb') as file:  
        model = pickle.load(file)

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
    plt.plot(fpr, tpr, marker='.', label=model, color='green')
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

    plt.plot(fpr, tpr, marker='.', label=model, color='purple')
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left');
    
    return None


def roc_curve_comp(X_test, y_test, model_names):

    '''
    Plots the ROC Curve for all listed models,
    '''
    
    plt.figure(figsize=(17,6)) 
    
    # ROC curve
    for m in model_names:
        
        with open(f'../models/{m}.pkl', 'rb') as file:  
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
        
        with open(f'../models/{m}.pkl', 'rb') as file:  
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


def learning_curve_comp(model_names, X_train, y_train):

    '''
    Plots a learning curve for all listed models.
    '''

    plt.figure(figsize = (13,6))
    
    colors = ['royalblue',
              'maroon', 
              'magenta', 
              'gold', 
              'limegreen'] 

    for i, x in zip(model_names, colors[:len(model_names)+1]):

        filename = f'../models/{i}.pkl'

        with open(filename, 'rb') as file:
            model = pickle.load(file)

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, 
                                                                              X_train, 
                                                                              y_train, 
                                                                              cv=5, 
                                                                              return_times=True,
                                                                              verbose=0) 

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.grid()
        plt.plot(train_sizes, train_scores_mean, "x-", color=x, label=f"{i[0:4]} Train")
        plt.plot(train_sizes, test_scores_mean, ".-", color=x, label=f"{i[0:4]} Test")
    
    plt.xlim([1000, 80000])
    plt.ylim([0.0, 1.2])
    plt.title(f'Learning Curve Comparison for {len(model_names)} Models')
    plt.xlabel('Training Samples')
    plt.ylabel('Score')
    plt.legend(title='Models', loc='lower right');        
        
    return None


## in progress
def visualize_plotpreds(model_name, v_train_data, X_test):
    
    filename = f'../models/{model_name}_model_{v_train_data}.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    preds = model.predict(X_test)
 
    sns.heatmap(preds.reshape((14,14)), vmin=0, vmax=.8).set_title(model_name)

    return None


## in progress
def make_gpro_friendly(raster_filepath):
    '''
    transforms a raster into format that can be imported into
    Google Earth Pro
    '''
    
    ds = gdal.Open(raster_filepath, 1)
    band = ds.GetRasterBand(1)
    colors = gdal.ColorTable()

    # set color for each value
    colors.SetColorEntry(0, (240, 247, 240)) 
    colors.SetColorEntry(1, (240, 247, 240))
    
    # set color table and color interpretation
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    # close and save file
    del band, ds
    
    return None