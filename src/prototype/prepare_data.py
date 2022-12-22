#! /usr/bin/env python3

import yaml
import hickle as hkl
import pickle
import pandas as pd
import numpy as np
import os
from natsort import natsorted
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random

# Load Data -- download from s3 via jupyter notebook
# Plot ID Labeling
# Plot IDs are numbered according to ceo survey
# the last three digits refer to the plot number and the first two digits refer to the survey
# for ex: 25th plot in ceo-plantations-train-v04.csv will be 04025.npy or 04025.hkl

def load_slope(idx, directory = '../data/train-slope/'):
    """
    Slope is stored as a 32 x 32 float32 array with border information.
    Needs to be converted to 1 x 14 x 14 array to match the labels w/ new axis.
    directory = '../data/train-slope/'
    """
    # Remove axes of length one with .squeeze (32 x 32 x 1)
    x = np.load(directory + str(idx) + '.npy').squeeze()
    original_shape = x.shape

    # slice out border information
    border_x = (x.shape[0] - 14) // 2
    border_y = (x.shape[1] - 14) // 2
    slope = x[border_x:-border_x, border_y:-border_y]

    # explicitly convert array to a row vector, adding axis (1 x 14 x 14)
    slope = slope[np.newaxis]
    
    #print(f'{idx} slope: {original_shape} -> {slope.shape}, {slope.dtype}')
    return slope


def load_s1(idx, directory = '../data/train-s1/'):
    """
    S1 is stored as a (12, 32, 32, 2) float64 array with border information.
    Needs to be converted from monthly mosaics to an annual median, 
    and to 14 x 14 x 2 to match labels. Dtype needs to be converted to float32.
    """
    
    s1 = hkl.load(directory + str(idx) + '.hkl')
    original_shape = s1.shape
    
    # get the median across flattened array
    if len(s1.shape) == 4:
        s1 = np.median(s1, axis = 0)
        
    # slice out border information
    border_x = (s1.shape[0] - 14) // 2
    border_y = (s1.shape[1] - 14) // 2
    s1 = s1[border_x:-border_x, border_y:-border_y]
   
    s1 = s1.astype(np.float32)
    
    #print(f'{idx} s1: {original_shape} -> {s1.shape}, {s1.dtype}')
    return s1


def load_s2(idx, directory = '../data/train-s2/'):
    
    """
    S2 is stored as a (12, 28, 28, 11) uint16 array. Remove the last axis index - the 
    date of the imagery. Convert monthly images to an 
    annual median. Remove the border to correspond to the labels.
    Convert to float32.
    
    """
    
    s2 = hkl.load(directory + str(idx) + '.hkl')
    original_shape = s2.shape
    
    # remove date of imagery (last axis)
    if s2.shape[-1] == 11:
        s2 = np.delete(s2, -1, -1)
    
    # convert monthly images to annual median
    if len(s2.shape) == 4:
        s2 = np.median(s2, axis = 0)
    
    # checks for floating datatype
    if not isinstance(s2.flat[0], np.floating):
        assert np.max(s2) > 1
        s2 = s2.astype(np.float32) / 65535
        assert np.max(s2) < 1
            
    # slice out border information
    border_x = (s2.shape[0] - 14) // 2
    border_y = (s2.shape[1] - 14) // 2
    s2 = s2[border_x:-border_x, border_y:-border_y].astype(np.float32)

    # print(f'{idx} s2: {original_shape} -> {s2.shape}, {s2.dtype}')
    return s2


def load_feats(idx, drop_prob, directory = '../data/train-features/'):
    '''
    Features are stored as a 14 x 14 x 65 float64 array. The last axis contains 
    the feature dimensions. Dtype needs to be converted to float32. The TML
    probability/prediction can optionally be dropped.
    
    Index 0 ([...,0]) is the tree cover prediction from the full TML model
    Index 1 - 33 are high level features
    Index 33 - 65 are low level features
    '''
    feats = hkl.load(directory + str(idx) + '.hkl').astype(np.float32)

    if drop_prob == True:
        feats = feats[..., :64]
        
    # print(f'{idx} feats: {feats.shape}, {feats.dtype}')
    return feats


def load_label(idx, directory = '../data/train-labels/'):
    '''
    The labels are stored as a binary 14 x 14 float64 array.
    Dtype needs to be converted to float32.
    '''
    labels = np.load(directory + str(idx) + '.npy').astype(np.float32)
    original_shape = labels.shape

    #print(f'{idx} labels: {labels.shape}, {labels.dtype}')
    return labels

# Create X and y variables

def make_sample(sample_shape, slope, s1, s2, feats):
    
    ''' 
    Defines dimensions and then combines slope, s1, s2 and TML features from a plot
    into a sample with shape (14, 14, 78)
    '''
    
    # define the last dimension of the array
    n_feats = 1 + s1.shape[-1] + s2.shape[-1] + feats.shape[-1]

    sample = np.empty((sample_shape[1], sample_shape[-1], n_feats))
    
    # populate empty array with each feature
    sample[..., 0] = slope
    sample[..., 1:3] = s1
    sample[..., 3:13] = s2
    sample[..., 13:] = feats
    
    return sample

def make_sample_nofeats(sample_shape, slope, s1, s2):
    
    ''' 
    Defines dimensions and then combines slope, s1 and s2 features from a plot
    into a sample with shape (14, 14, 13). 
    '''
    
    # define the last dimension of the array
    n_feats = 1 + s1.shape[-1] + s2.shape[-1] 

    sample = np.empty((sample_shape[1], sample_shape[-1], n_feats))
    
    # populate empty array with each feature
    sample[..., 0] = slope
    sample[..., 1:3] = s1
    sample[..., 3:13] = s2
    
    return sample

def create_xy(sample_shape, v_train_data, drop_prob, drop_feats, verbose=False):
    '''
    Creates an empty array for x and y based on the training data set
    then creates samples and labels by loading data by plot ID. Removes ids where 
    there is no cloud-free imagery available.
    Combines all samples into a single array as input to the model.
    Also returns a baseline accuracy score (indicating class imbalance)
    '''
    # use CEO csv to gather plot id numbers
    if len(v_train_data) == 1:
        df = pd.read_csv(f'../data/ceo-plantations-train-{v_train_data[0]}.csv')
        plot_ids = df.PLOT_FNAME.drop_duplicates().tolist()

    elif len(v_train_data) > 1:
        plot_ids = []
        for i in v_train_data:
            df = pd.read_csv(f'../data/ceo-plantations-train-{i}.csv')
            plot_ids = plot_ids + df.PLOT_FNAME.drop_duplicates().tolist()
    
    # if the plot_ids do not have 5 digits, change to str and add leading 0
    plot_ids = [str(item).zfill(5) if len(str(item)) < 5 else item for item in plot_ids]

    # check and remove any plot ids where there are no cloud free images (no feats or s2)
    for plot in plot_ids:
        if not os.path.exists(f'../data/train-s2/{plot}.hkl') and not os.path.exists(f'../data/train-features/{plot}.hkl'):
            print(f'Plot id {plot} has no cloud free imagery and will be removed.')
            plot_ids.remove(plot)

    # manually remove this plot that was getting skipped for unknown reason
    plot_ids.remove('04008')
    # plot_ids.remove('08182')
    # plot_ids.remove('09168')
    # plot_ids.remove('09224')

    if verbose:
        print(f'Training data includes {len(plot_ids)} plot ids.')

    # create empty x and y array based on number of plots (dropping TML probability changes dimensions from 78 -> 77)
    n_samples = len(plot_ids)
    y_all = np.empty(shape=(n_samples, 14, 14))

    if drop_prob:
        x_all = np.empty(shape=(n_samples, 14, 14, 77))
    elif drop_feats:
        x_all = np.empty(shape=(n_samples, 14, 14, 13))
    else:
        x_all = np.empty(shape=(n_samples, 14, 14, 78))
    
    for num, plot in enumerate(plot_ids):

        if drop_feats:
            X = make_sample_nofeats(sample_shape, load_slope(plot), load_s1(plot), load_s2(plot))
            y = load_label(plot)
            x_all[num] = X
            y_all[num] = y

        else:
            # at index i, load and create the sample, then append to empty array
            X = make_sample(sample_shape, load_slope(plot), load_s1(plot), load_s2(plot), load_feats(plot, drop_prob))
            y = load_label(plot)
            x_all[num] = X
            y_all[num] = y

        if verbose:
            print(f'Sample: {num}')
            print(f'Features: {X.shape}, Labels: {y.shape}')
        
    # check class balance and baseline accuracy
    labels, counts = np.unique(y_all, return_counts=True)
    print(f'Baseline: {round(counts[0] / (counts[0] + counts[1]), 3)}')

    return x_all, y_all



def reshape_and_scale_manual(X, y, v_train_data, verbose=False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)
    start_min, start_max = X_train.min(), X_train.max()

    if verbose:
        print(f'X_train: {X_train.shape} X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}')

    # standardize train/test data 
    min_all = []
    max_all = []

    for band in range(0, X_train.shape[-1]):
        
        mins = np.percentile(X_train[..., band], 1)
        maxs = np.percentile(X_train[..., band], 99)
        
        if maxs > mins:
            
            # clip values in each band based on min/max of training dataset
            X_train[..., band] = np.clip(X_train[..., band], mins, maxs)
            X_test[..., band] = np.clip(X_test[..., band], mins, maxs)

            #calculate standardized data
            midrange = (maxs + mins) / 2
            rng = maxs - mins
            X_train_std = (X_train[..., band] - midrange) / (rng / 2)
            X_test_std = (X_test[..., band] - midrange) / (rng / 2)

            # update each band in X_train and X_test to hold standardized data
            X_train[..., band] = X_train_std
            X_test[..., band] = X_test_std
            end_min, end_max = X_train.min(), X_train.max()
            
            min_all.append(mins)
            max_all.append(maxs)
        else:
            pass
    
    # save mins and maxs 
    np.save(f'../data/mins_{v_train_data}', min_all)
    np.save(f'../data/maxs_{v_train_data}', max_all)

    ## reshape
    X_train_ss = np.reshape(X_train, (np.prod(X_train.shape[:-1]), X_train.shape[-1]))
    X_test_ss = np.reshape(X_test, (np.prod(X_test.shape[:-1]), X_test.shape[-1]))
    y_train = np.reshape(y_train, (np.prod(y_train.shape[:])))
    y_test = np.reshape(y_test, (np.prod(y_test.shape[:])))
    if verbose:
        print(f'Reshaped X_train: {X_train_ss.shape} X_test: {X_test_ss.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}')
        print(f"The data was scaled to: Min {start_min} -> {end_min}, Max {start_max} -> {end_max}")

    return X_train_ss, X_test_ss, y_train, y_test
