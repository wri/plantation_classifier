#! /usr/bin/env python3

import hickle as hkl
import pickle
import pandas as pd
import numpy as np
import os
import seaborn as sns
import prepare_data as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from prepare_data import convert_to_db

## requires make sample and training data


def load_large_feats(shape, directory='../data/large-features/', verbose=False):

    '''
    Input is the location of the raw data for larger map.
    Imports data and updates datatype and shape
    Returns sample with shape (500, 500, 78)
    '''
    
    # load slope
    slope = hkl.load(directory + 'slope.hkl').squeeze()
    slope_orig = slope.shape
    slope = slope[np.newaxis]
    
    # load s1
    s1 = hkl.load(directory + 's1.hkl')
    s1_orig = s1.shape
    if len(s1.shape) == 4:
        s1 = np.median(s1, axis = 0)
    s1 = s1.astype(np.float32)
    s1[..., -1] = convert_to_db(s1[..., -1], 22)
    s1[..., -2] = convert_to_db(s1[..., -2], 22)
    
    # load s2
    s2 = hkl.load(directory + 's2.hkl')
    s2_orig = s2.shape
    if s2.shape[-1] == 11:
        s2 = np.delete(s2, -1, -1)
    if not isinstance(s2.flat[0], np.floating):
        assert np.max(s2) > 1
        s2 = s2.astype(np.float32) / 65535
        assert np.max(s2) < 1
    if len(s2.shape) == 4:
        s2 = np.median(s2, axis = 0)
        
    # load features
    feats = hkl.load(directory + 'features.hkl').astype(np.float32)

    if verbose:
        print(f's1: {s1_orig} -> {s1.shape}, {s1.dtype}')
        print(f's2: {s2_orig} -> {s2.shape}, {s2.dtype}')
        print(f'slope: {slope_orig} -> {slope.shape}, {slope.dtype}')
      
        
    # create the sample and reshape -- input shape should be (500, 500)
    largefeats = pd.make_sample(shape, slope, s1, s2, feats)
    
    return largefeats


def reshape_and_scale(v_train_data: list, unseen, verbose=False):
    
    '''
    V_training_data: list of training data version
    unseen: new sample 

    Takes in a large sample with dimensions (500, 500, 78) and 
    reshapes to (250,000, 78), then applies scaling using the training data
    '''
    # prepare original training data for vectorizer
    X, y = pd.create_xy((14,14), v_train_data, drop_prob=False, drop_feats=False, verbose=False)

    # train test split before reshaping to ensure plot is not mixed samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)

    # reshape arrays (only Xtrain and unseen)
    X_train_reshaped = np.reshape(X_train, (np.prod(X_train.shape[:-1]), X_train.shape[-1]))
    unseen_reshaped = np.reshape(unseen, (np.prod(unseen.shape[:-1]), unseen.shape[-1]))
    if verbose:
        print(f'Xtrain Original: {X_train.shape} Xtrain Reshaped: {X_train_reshaped.shape}')
        print(f'Unseen Original: {unseen.shape} Unseen Reshaped: {unseen_reshaped.shape}')

    # apply standardization on a  copy
    X_train_ss = X_train_reshaped.copy()
    unseen_ss = unseen_reshaped.copy()

    # scaler = StandardScaler()
    # X_train_ss = scaler.fit_transform(X_train_ss)
    # unseen_ss = scaler.transform(unseen_ss)
    # if verbose:
    #     print(f'Scaled to {np.min(X_train_ss)}, {np.max(X_train_ss)}')
    #     print(f'Scaled to {np.min(unseen_ss)}, {np.max(unseen_ss)}')
    
    return unseen_ss


def visualize_large_feats(model_name, v_train_data, largefeats):
    
    filename = f'../models/{model_name}_model_{v_train_data}.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    preds = model.predict(largefeats)
 
    sns.heatmap(preds.reshape((500, 500)), vmin=0, vmax=.8).set_title(model_name)

    return None