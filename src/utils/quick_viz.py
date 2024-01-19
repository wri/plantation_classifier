#!/usr/bin/env python

import seaborn as sns
import matplotlib.pyplot as plt
import rasterio as rs
import hickle as hkl
import numpy as np
import pickle

# from sklearn.model_selection import learning_curve
# from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

# Base
# └── basic visualizations
#     ├── heatmaps
#     │   ├── multiply.py
#     │   ├── divide.py
#     ├── histograms
#     ├── confusion matrix

def heat_multiplot(matrices, cbarmin, cbarmax,  nrows = 13, ncols = 6):
    '''
    Type: Seaborn heatmap
    Purpose: Create a multiplot of heatmaps from a collection of matrices.

    Parameters:
    - matrices (array-like): A collection of matrices to be visualized as heatmaps.
    - cbarmin (float): The minimum value for the color bar scale.
    - cbarmax (float): The maximum value for the color bar scale.
    - nrows (int, optional): Number of rows for the subplot grid. Default is 13.
    - ncols (int, optional): Number of columns for the subplot grid. Default is 6.

    This function creates a multiplot of heatmaps from a collection of matrices. This function is most
    helpful if you want to visualize 3D arrays. It arranges the heatmaps in a grid with the specified 
    number of rows and columns. The color scale for the heatmaps is defined by the `cbarmin` 
    and `cbarmax` parameters. Each heatmap is displayed within its own subplot.

    Returns:
    None
    
    '''
    fig, axs = plt.subplots(ncols = ncols, nrows = nrows)
    fig.set_size_inches(18, 3.25*nrows)
    
    # create a list of indices from 0 to nrows*ncols
    to_iter = [[x for x in range(i, i + ncols + 1)] for i in range(0, nrows*ncols, ncols)]
    counter = 0
    
    #
    for r in range(1, nrows + 1):
        min_i = min(to_iter[r-1])
        max_i = max(to_iter[r-1])
        
        for i in range(ncols):
            sns.heatmap(data = matrices[..., counter], 
                        ax = axs[r - 1, i], 
                        cbar = True, 
                        vmin = cbarmin, # this could also be min_i
                        vmax = cbarmax, # this could also be max_i
                        cmap = sns.color_palette("viridis", as_cmap=True))
            axs[r - 1, i].set_xlabel("")
            axs[r - 1, i].set_ylabel("")
            axs[r - 1, i].set_yticks([])
            axs[r - 1, i].set_xticks([])
            counter += 1
        
    plt.show
    return None

def heat_compare_preds(location: str, tile_idx_a: tuple, tile_idx_b: tuple):
    '''
    Type: Seaborn heatmap
    Purpose: Compare and visualize multi-class predictions for two specific tiles in a given location 
    using heatmaps.

    Parameters:
    - location (str): The country.
    - tile_idx_a (tuple): The coordinates (x, y) of the first tile to compare.
    - tile_idx_b (tuple): The coordinates (x, y) of the second tile to compare.
    - title (str): The title to be displayed for the comparison plot.

    This function loads prediction data for two tiles specified by their coordinates (tile_idx_a and tile_idx_b)
    from the specified location directory and creates side-by-side heatmaps to visualize the predictions. The
    heatmaps display values in the range [0, 1, 2], and each tile's heatmap is shown in a separate subplot.

    Returns:
    None
    
    '''
    x_a, y_a = tile_idx_a[0], tile_idx_a[1]
    x_b, y_b = tile_idx_b[0], tile_idx_b[1]
    preds_a = rs.open(f'../tmp/{location}/preds/{str(x_a)}X{str(y_a)}Y_preds.tif').read(1)
    preds_b = rs.open(f'../tmp/{location}/preds/{str(x_b)}X{str(y_b)}Y_preds.tif').read(1)
    

    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    sns.heatmap(preds_a, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [0, 1, 2]}, 
                vmin=0, vmax=2).set_title('Tile: ' + str(tile_idx_a))
    plt.subplot(1,2,2)
    sns.heatmap(preds_b, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [0, 1, 2]}, 
                vmin=0, vmax=2).set_title('Tile: ' + str(tile_idx_b));

    return None

def hist_compare_s2(location: str, tile_idx_a: tuple, tile_idx_b: tuple, title:str, tile_idx_c: tuple = None):
    '''
    Type: Matplotlib histogram
    Purpose: Compare and visualize histograms of Sentinel-2 data for two specific tiles in a given location.

    Parameters:
    - location (str): The country.
    - tile_idx_a (tuple): The coordinates (x, y) of the first tile to compare.
    - tile_idx_b (tuple): The coordinates (x, y) of the second tile to compare.
    - title (str): The title for the histogram comparison plot, describing the area.

    This function loads analysis ready data for two tiles specified by their coordinates (tile_idx_a and tile_idx_b). 
    It then indexes the array to creates histograms of Sentinel-2 data and displays 
    them in a single plot for visual comparison.

    Returns:
    None
    
    '''
    x_a, y_a = tile_idx_a[0], tile_idx_a[1]
    x_b, y_b = tile_idx_b[0], tile_idx_b[1]
    ard_a = hkl.load(f'../tmp/{location}/{str(x_a)}/{str(y_a)}/ard/{str(x_a)}X{str(y_a)}Y_ard.hkl')
    ard_b = hkl.load(f'../tmp/{location}/{str(x_b)}/{str(y_b)}/ard/{str(x_b)}X{str(y_b)}Y_ard.hkl')
    
    s2_a = ard_a[..., 0:10]
    s2_b = ard_b[..., 0:10]
    
    plt.figure(figsize=(6,4))
    binwidth = .01
    min = s2_a.min()
    max = s2_a.max()

    # this asks for 33 binds between .01 and .6 -- np.arange(min, max + binwidth, binwidth)
    plt.hist(s2_a.flatten(), alpha=0.5, label=str(tile_idx_a), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
    plt.hist(s2_b.flatten(), alpha=0.3, label=str(tile_idx_b), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
    plt.xlim(0.0, 0.6)
    plt.xticks(np.arange(0.0, 0.6, 0.1))
    plt.title(title)
    plt.legend();

    if tile_idx_c is not None:
        x_c, y_c = tile_idx_c[0], tile_idx_c[1]
        ard_c = hkl.load(f'../tmp/{location}/{str(x_c)}/{str(y_c)}/ard/{str(x_c)}X{str(y_c)}Y_ard.hkl')
        s2_c = ard_c[..., 0:10]
        plt.hist(s2_c.flatten(), alpha=0.3, label=str(tile_idx_c), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
        plt.legend();

    return None

def hist_compare_s2_byband(location: str, tile_idx_a: tuple, tile_idx_b: tuple, title:str, tile_idx_c: tuple = None):
    '''
    Each s2 band is plot 
    
    '''
    x_a, y_a = tile_idx_a[0], tile_idx_a[1]
    x_b, y_b = tile_idx_b[0], tile_idx_b[1]
    ard_a = hkl.load(f'../tmp/{location}/{str(x_a)}/{str(y_a)}/ard/{str(x_a)}X{str(y_a)}Y_ard.hkl')
    ard_b = hkl.load(f'../tmp/{location}/{str(x_b)}/{str(y_b)}/ard/{str(x_b)}X{str(y_b)}Y_ard.hkl')
    
    s2_a = ard_a[..., 0:10]
    s2_b = ard_b[..., 0:10]

    plt.figure(figsize=(20,20))
    binwidth = .01
    min = s2_a.min()
    max = s2_a.max()
    band_counter = 0
    
    for i in range(1, 11):
        plt.subplot(4,3,i)
        plt.hist(s2_a[..., band_counter].flatten(), alpha=0.5, label=str(tile_idx_a), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
        plt.hist(s2_b[..., band_counter].flatten(), alpha=0.3, label=str(tile_idx_b), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
        plt.xlim(0.0, 0.5)
        plt.xticks(np.arange(0.0, 0.5, 0.1))
        plt.title(title + f' Band {str(band_counter)}')
        #plt.legend();

        if tile_idx_c is not None:
            x_c, y_c = tile_idx_c[0], tile_idx_c[1]
            ard_c = hkl.load(f'../tmp/{location}/{str(x_c)}/{str(y_c)}/ard/{str(x_c)}X{str(y_c)}Y_ard.hkl')
            s2_c = ard_c[..., 0:10]
            plt.hist(s2_c[..., band_counter].flatten(), alpha=0.3, label=str(tile_idx_c), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
        
        band_counter += 1

    return None




def heat_compare_arrays(arr_a, arr_b, title_a:str, title_b:str):
    '''
    Type: Seaborn heatmap
    Purpose: Compare and visualize two files (could be s2 data, ARD, feats, etc.)
    
    '''

    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    sns.heatmap(arr_a, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [arr_a.min(), arr_a.max()]}).set_title(title_a)
        
    plt.subplot(1,2,2)
    sns.heatmap(arr_b, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [arr_b.min(), arr_b.max()]}).set_title(title_b);

    return None



def cm_roc_pr(model, y_test, pred, probs_pos):

    ''' 
    Visualize the performance of a classification model using a Confusion Matrix, 
    ROC Curve, and Precision-Recall Curve.

    Parameters:
    - model: The trained classification model.
    - y_test: True labels of the test set.
    - pred: Predicted labels of the test set.
    - probs_pos: Probability of the positive class for each sample in the test set.

    Note:
    This function requires the scikit-learn library for confusion matrix visualization
    and matplotlib for creating ROC and Precision-Recall curves.
    '''
    
    with open(f'../models/{model}.pkl', 'rb') as file:  
        model = pickle.load(file)

     # Calculate and plot CM
    cm = confusion_matrix(y_test, pred, labels=model.classes_)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot();

    # Calculate and plot ROC AUC 
    fpr, tpr, thresholds = roc_curve(y_test, probs_pos)

    plt.figure(figsize=(17,6)) 

    plt.subplot(1,2,1)
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=model, color='green')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right');

    
    # Calculate and plot precision-recall curve and no skill
    fpr, tpr, thresholds = precision_recall_curve(y_test, probs_pos)
    no_skill = len(y_test[y_test == 1]) / len(y_test)

    plt.subplot(1,2,2)
    plt.plot([0,1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=model, color='purple')
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left');
    
    return None



def roc_curve_comp(X_test, y_test, model_names):

    '''
    Plot ROC Curves for multiple classification models.

    Parameters:
    - X_test: Testing features.
    - y_test: True labels of the test set.
    - model_names: List of models to be plotted and compared.

    Note:
    This function requires scikit-learn for calculating ROC curves.
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


def learning_curve_comp(model_names, X_train, y_train, x_max):

    '''
    Plot learning curves to compare the performance of machine learning models.

    Parameters:
    - model_names: List of model names to be compared.
    - X_train: Training features.
    - y_train: True labels of the training set.
    - x_max: Maximum number of training samples to display on the x-axis.

    Note: This function requires scikit-learn for learning curve computation
    '''
    colors = ['royalblue',
              'maroon', 
              'magenta', 
              'gold', 
              'limegreen'] 
    
    plt.figure(figsize = (13,6))
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
    
    plt.xlim([1000, x_max])
    plt.ylim([0.0, 1.2])
    plt.title(f'Learning Curve Comparison for {len(model_names)} Models')
    plt.xlabel('Training Samples')
    plt.ylabel('Score')
    plt.legend(title='Models', loc='lower right');        
        
    return None