#! /usr/bin/env python3

import hickle as hkl
import pickle
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import validate_io as validate
import slow_glcm as slow_txt
import fast_glcm as fast_txt
from tqdm import tqdm

### Plot ID Labeling ###
# Plot IDs are numbered according to ceo survey
# the last three digits refer to the plot number and the first two digits refer to the survey
# for ex: 25th plot in ceo-plantations-train-v04.csv will be 04025.npy or 04025.hkl

def convert_to_db(x: np.ndarray, min_db: int) -> np.ndarray:
    """ 
    Converts Sentinel 1 unitless backscatter coefficient
    to decible with a min_db lower threshold

    Background: S1 is required to be processed to equivalent backscatter coefficient
    images in decibels (dB) scale. This backscatter coefficient represents the target 
    backscattering area (radar cross-section) per unit ground area. 
    It is required to be converted into dB as it can vary by several orders of magnitude. 
    It measures whether the surface backscatters from the incident microwave radiation 
    are preferentially away from the SAR sensor dB < 0) or towards the SAR sensor dB > 0).
    
    Parameters:
        x (np.ndarray): unitless backscatter (T, X, Y, B) array
        min_db (int): integer from -50 to 0
            (-22 db is the lower limit of sensitivity for s1)

    Returns:
        x (np.ndarray): db backscatter (T, X, Y, B) array
    
    (T, X, Y, B): time, x dimension, y dimension, band
    """
    # converts the array to decibel
    x = 10 * np.log10(x + 1/65535)
    x[x < -min_db] = -min_db
    x = (x + min_db) / min_db

    # return array clipped to values between 0-1
    x = np.clip(x, 0, 1)
    
    return x


def load_slope(idx, directory = '../data/train-slope/'):
    """
    Slope is stored as a 32 x 32 float32 array with border information.
    Needs to be converted to 1 x 14 x 14 array to match the labels w/ new axis.
    directory = '../data/train-slope/'
    """
    # Remove axes of length one with .squeeze (32 x 32 x 1)
    x = np.load(directory + str(idx) + '.npy').squeeze()

    # slice out border information
    border_x = (x.shape[0] - 14) // 2
    border_y = (x.shape[1] - 14) // 2
    slope = x[border_x:-border_x, border_y:-border_y]

    # explicitly convert array to a row vector, adding axis (1 x 14 x 14)
    slope = slope[np.newaxis]
    
    return slope


def load_s1(idx, directory = '../data/train-s1/'):
    """
    S1 is stored as a (12, 32, 32, 2) float64 array with border information.
    Needs to be converted from monthly mosaics to an annual median, 
    and remove border information to match labels. 
    Dtype needs to be converted to float32.
    """

    # since s1 is a float64, can't use to_float32() 
    
    s1 = hkl.load(directory + str(idx) + '.hkl')
    
    # convert to decible
    s1[..., -1] = convert_to_db(s1[..., -1], 22)
    s1[..., -2] = convert_to_db(s1[..., -2], 22)

    # get the median across flattened array
    if len(s1.shape) == 4:
        s1 = np.median(s1, axis = 0, overwrite_input=True)

    # just convert (removed /65535) keep here
    s1 = np.float32(s1)

    # slice out border information (32, 32, 2) -> (14, 14, 2)
    border_x = (s1.shape[0] - 14) // 2
    border_y = (s1.shape[1] - 14) // 2
    s1 = s1[border_x:-border_x, border_y:-border_y]

    return s1


def load_s2(idx, directory = '../data/train-s2/'):
    
    """
    S2 is stored as a (12, 28, 28, 11) uint16 array. 
    Remove the last axis index - the date of the imagery. 
    Convert monthly images to an annual median. 
    Remove the border to correspond to the labels.
    Convert to float32.
    """
    
    s2 = hkl.load(directory + str(idx) + '.hkl')

    # remove date of imagery (last axis)
    if s2.shape[-1] == 11:
        s2 = np.delete(s2, -1, -1)

    # checks for floating datatype, if not converts to float32
    # TODO check if this is the same func as in deply pipeline
    # if so consider moving to utils
    if not isinstance(s2.flat[0], np.floating):
        assert np.max(s2) > 1
        s2 = s2.astype(np.float32) / 65535
        assert np.max(s2) < 1
 
    # convert monthly images to annual median
    if len(s2.shape) == 4:
        s2 = np.median(s2, axis = 0, overwrite_input=True)

    # slice out border information
    border_x = (s2.shape[0] - 14) // 2
    border_y = (s2.shape[1] - 14) // 2
    s2 = s2[border_x:-border_x, border_y:-border_y].astype(np.float32)

    return s2

def load_ard(idx, directory = '../data/train-ard/'):
    '''
    Analysis ready data is stored as (12, 28, 28, 17)
    Get the median for the year, cut out border information
    and drop spectral indices
    (28, 28, 17)
    (28, 28, 13)
    (14, 14, 13)
    '''
    ard = np.load(directory + str(idx) + '.npy')

    # convert monthly images to annual median
    if len(ard.shape) == 4:
        ard = np.median(ard, axis = 0, overwrite_input=True)

    ard = ard[..., :13]
    
    # slice out border information
    border_x = (ard.shape[0] - 14) // 2
    border_y = (ard.shape[1] - 14) // 2
    ard = ard[border_x:-border_x, border_y:-border_y].astype(np.float32)

    return ard


def load_ttc(idx, directory = '../data/train-features-ckpt-2023-02-09/'):
    '''
    Features are stored as a 14 x 14 x 65 float64 array. The last axis contains 
    the feature dimensions. Dtype needs to be converted to float32. The TML
    probability/prediction can optionally be dropped.

    ## Update per 2/13/23
    Features range from -infinity to +infinity
    and must be clipped to be consistent with the deployed features.
    
    Index 0 ([...,0]) is the tree cover prediction from the full TML model
    Index 1 - 33 are high level features
    Index 33 - 65 are low level features
    '''
    feats = hkl.load(directory + str(idx) + '.hkl')

    # clip all features after indx 0 to specific vals
    feats[..., 1:] = np.clip(feats[..., 1:], a_min=-32.768, a_max=32.767)

    feats = feats.astype(np.float32)

    return feats


def load_label(idx, binary, directory = '../data/train-labels/'):
    '''
    The labels are stored as a binary 14 x 14 float64 array.
    Unless they are stored as (196,) and need to be reshaped.
    Dtype needs to be converted to float32.
    '''
    labels_raw = np.load(directory + str(idx) + '.npy')

    if len(labels_raw.shape) == 1:
        labels_raw = labels_raw.reshape(14, 14)

    # makes sure that a binary classification exercise updates
    # any multiclass labels (this is just converting AF label (2) to 1)
    if binary:
        labels = labels_raw.copy()
        labels[labels_raw == 2] = 1
        labels = labels.astype(np.float32)
        
    else:
        labels = labels_raw.astype(np.float32)
    
    return labels


def load_txt(idx, directory = '../data/train-ard/'):
    
    '''
    Loads raw s2 data and preprocesses
    in order to extract texture features for RGB and NIR 
    bands. Outputs the texture analysis as a (14, 14, 16) 
    array as a float32 array.
    '''
    if os.path.exists(f'../data/train-texture/{idx}_ard.npy'):
        output = np.load(f'../data/train-texture/{idx}_ard.npy')
    
    else: 
        # #print('Calculating GLCM texture features...')
        # s2 = hkl.load(directory + str(idx) + '.hkl')
        
        # # remove date of imagery (last axis)
        # if s2.shape[-1] == 11:
        #     s2 = np.delete(s2, -1, -1)

        # # convert monthly images to annual median
        # if len(s2.shape) == 4:
        #     s2 = np.median(s2, axis = 0, overwrite_input=True)

        # this has to be done after median
        # doesnt work if just calling .astype(np.uint8)
        #s2 = ((s2.astype(np.float32) / 65535) * 255).astype(np.uint8)

        # prepare s2 from ard without removing border info
        ard = np.load(directory + str(idx) + '.npy')
        if len(ard.shape) == 4:
            ard = np.median(ard, axis = 0, overwrite_input=True)
        s2 = ard[..., 0:10]
        s2 = ((s2.astype(np.float32) / 65535) * 255).astype(np.uint8)
        
        blue = s2[..., 0]
        green = s2[..., 1]
        red = s2[..., 2]
        nir = s2[..., 3]
        output = np.zeros((14, 14, 16))
        
        output[..., 0:4] = slow_txt.extract_texture(blue)
        output[..., 4:8] = slow_txt.extract_texture(green)
        output[..., 8:12] = slow_txt.extract_texture(red)
        output[..., 12:16] = slow_txt.extract_texture(nir)

        # save the output
        np.save(f'../data/train-texture/{idx}_ard.npy', output)
    
    return output.astype(np.float32)


def make_sample_OLDDONTUSE(sample_shape, slope, s1, s2, txt, ttc, feature_select):
    
    ''' 
    Defines dimensions and then combines slope, s1, s2, TML features and 
    texture features from a plot into a sample with shape (14, 14, 94)
    Feature select is a list of features that will be used, otherwise empty list
    '''
    # now filter to select features if arg provided
    # squeeze extra axis that is added (14,14,1,15) -> (14,14,15)
    feats = np.zeros((sample_shape[0], sample_shape[1], ttc.shape[-1] + txt.shape[-1]), dtype=np.float32)
    feats[..., :65] = ttc
    feats[..., 65:] = txt
    if len(feature_select) > 0:
        feats = np.squeeze(feats[:, :, [feature_select]])

    # define the last dimension of the array
    n_feats = 1 + s1.shape[-1] + s2.shape[-1] + feats.shape[-1] 

    sample = np.empty((sample_shape[0], sample_shape[1], n_feats))

    # populate empty array with each feature
    sample[..., 0] = slope
    sample[..., 1:3] = s1
    sample[..., 3:13] = s2
    sample[..., 13:] = feats

    return sample


def make_sample(sample_shape, ard, txt, ttc, feature_select):

    # prepare the feats (this is dont first bc of feature selection)
    # squeeze extra axis that is added (14,14,1,15) -> (14,14,15)
    feats = np.zeros((sample_shape[0], sample_shape[1], ttc.shape[-1] + txt.shape[-1]), dtype=np.float32)
    feats[..., :ttc.shape[-1]] = ttc
    feats[..., ttc.shape[-1]:] = txt
    if len(feature_select) > 0:
        feats = np.squeeze(feats[:, :, [feature_select]])

    # create empty sample array
    n_feats = ard.shape[-1] + feats.shape[-1] 
    sample = np.zeros((ard.shape[0], ard.shape[1], n_feats), dtype=np.float32)

    # populate empty array with each feature
    # order: s2, dem, s1, ttc, txt
    sample[..., 0:10] = ard[..., 0:10]
    sample[..., 10:11] = ard[..., 10:11]
    sample[..., 11:13] = ard[..., 11:13]
    sample[..., 13:] = feats

    return sample


def gather_plot_ids(v_train_data):
    '''
    Creates a list of plot ids to process from collect earth surveys 
    with multi-class labels (0, 1, 2, 255). Drops all plots with 
    "unknown" labels and plots w/o s2 imagery. Returns list of plot_ids.
    TODO: how will cloudfree images be identified with ARD?
    '''

    # use CEO csv to gather plot id numbers
    plot_ids = []
    no_labels = []

    for i in v_train_data:

        # for each training data survey, drop all unknown labels
        df = pd.read_csv(f'../data/ceo-plantations-train-{i}.csv')

        # assert unknown labels are always a full 14x14 (196 points) of unknowns
        unknowns = df[df.PLANTATION == 255]
        no_labels.extend(sorted(list(set(unknowns.PLOT_FNAME))))
        for plot in set(list(unknowns.PLOT_ID)):
            assert len(unknowns[unknowns.PLOT_ID == plot]) == 196,\
            f'{plot} has {len(unknowns[unknowns.PLOT_ID == plot])}/196 points labeled unknown.'

        # drop unknowns and add to full list
        df_new = df.drop(df[df.PLANTATION == 255].index)
        plot_ids += df_new.PLOT_FNAME.drop_duplicates().tolist()

    # add leading 0 to plot_ids that do not have 5 digits
    plot_ids = [str(item).zfill(5) if len(str(item)) < 5 else str(item) for item in plot_ids]
        
    # remove any plot ids where there are no cloud free images (no s2 hkl file)
    #final_plots = [plot for plot in plot_ids if os.path.exists(f'../data/train-s2/{plot}.hkl')]
    #no_cloudfree = [plot for plot in plot_ids if plot not in final_plots]

    ### TO BE CONFIRMED for ARD
    final_plots = [plot for plot in plot_ids if os.path.exists(f'../data/train-ard/{plot}.npy')]

    print(f'{len(no_labels)} plots labeled "unknown" were dropped: {no_labels}')
    #print(f'{len(no_cloudfree)} plots had no cloud free imagery: {no_cloudfree}')
    print(f'Training data includes {len(final_plots)} plots.')

    return final_plots


def create_xy_OLDDONTUSE(v_train_data, binary, drop_feats, feature_select, verbose=False):
    '''
    Gathers training data plots from collect earth surveys (v1, v2, v3, etc)
    and loads data to create a sample for each plot. Removes ids where there is no
    cloud-free imagery or "unknown" labels. Option to process binary or multiclass
    labels.
    Combines samples as X and loads labels as y for input to the model. 
    Returns baseline accuracy score?

    TODO: finish documentation

    v_train_data:
    drop_feats:
    convert_binary:
    
    '''
    

    plot_ids = gather_plot_ids(v_train_data)
    print(f'Training data includes {len(plot_ids)} plots.')

    # create empty x and y array based on number of plots (dropping TML probability changes dimensions from 78 -> 77)
    sample_shape = (14, 14)
    n_samples = len(plot_ids)
    y_all = np.zeros(shape=(n_samples, 14, 14))

    if drop_feats:
        x_all = np.zeros(shape=(n_samples, 14, 14, 13))
    elif len(feature_select) > 0:
        x_all = np.zeros(shape=(n_samples, 14, 14, 13 + len(feature_select)))
    else:
        x_all = np.zeros(shape=(n_samples, 14, 14, 94))

    for num, plot in enumerate(tqdm(plot_ids)):

        if drop_feats:
            slope = load_slope(plot)
            s1 = load_s1(plot)
            s2 = load_s2(plot)
            X = make_sample_nofeats(sample_shape, slope, s1, s2)
            y = load_label(plot, binary)
            x_all[num] = X
            y_all[num] = y

        else:
            slope = load_slope(plot)
            s1 = load_s1(plot)
            s2 = load_s2(plot)
            ttc = load_ttc(plot)
            txt = load_txt(plot)
            validate.train_output_range_dtype(slope, s1, s2, ttc, feature_select)
            X = make_sample(sample_shape, slope, s1, s2, txt, ttc, feature_select)
            y = load_label(plot, binary)
            x_all[num] = X
            y_all[num] = y

            # clean up memory
            del slope, s1, s2, ttc, txt, X, y

        if verbose:
            print(f'Sample: {num}')
            print(f'Features: {X.shape}, Labels: {y.shape}')
        
    # check class balance 
    labels, counts = np.unique(y_all, return_counts=True)
    print(f'Class count {dict(zip(labels, counts))}')

    return x_all, y_all


def create_xy(v_train_data, binary, feature_select, sample_shape=(14, 14), verbose=False):
    '''
    Gathers training data plots from collect earth surveys (v1, v2, v3, etc)
    and loads data to create a sample for each plot. Removes ids where there is no
    cloud-free imagery or "unknown" labels. Option to process binary or multiclass
    labels.
    Combines samples as X and loads labels as y for input to the model. 
    Returns baseline accuracy score?



    TODO: finish documentation
    '''
    
    plot_ids = gather_plot_ids(v_train_data)
    
    if len(feature_select) > 0:
        n_feats = 13 + len(feature_select)
    else:
        n_feats = 94
        
    # create empty x and y array based on number of plots 
    # x.shape is (plots, 14, 14, n_feats) y.shape is (plots, 14, 14)
    n_samples = len(plot_ids)
    y_all = np.zeros(shape=(n_samples, sample_shape[0], sample_shape[1]))
    x_all = np.zeros(shape=(n_samples, sample_shape[0], sample_shape[1], n_feats))

    for num, plot in enumerate(tqdm(plot_ids)):
        ard = load_ard(plot)
        ttc = load_ttc(plot)
        txt = load_txt(plot) 
        validate.train_output_range_dtype(ard[...,10], ard[...,11:13], ard[...,0:10], ttc, feature_select) 
        X = make_sample(sample_shape, ard, txt, ttc, feature_select)
        y = load_label(plot, binary)
        x_all[num] = X
        y_all[num] = y

        # clean up memory
        del ard, ttc, txt, X, y

    if verbose:
        print(f'Sample: {num}')
        print(f'Features: {X.shape}, Labels: {y.shape}')
        
    # check class balance 
    labels, counts = np.unique(y_all, return_counts=True)
    print(f'Class count {dict(zip(labels, counts))}')

    return x_all, y_all


def reshape_arr(arr):
    '''
    Reshapes a 4D array (X) to 2D and a 3D array (y) to 1D for input into a 
    machine learning model.

    Parameters:
    - arr (array-like): Input array to be reshaped. For X, it is assumed to have 
    shape (plots, 14, 14, n_feats), and for y, it is assumed to have shape (plots, 14, 14).

    Returns:
    - reshaped (array-like): Reshaped array with dimensions suitable for machine learning model input.
    '''
    if len(arr.shape) == 4:
        reshaped = np.reshape(arr, (np.prod(arr.shape[:-1]), arr.shape[-1]))
    else:
        reshaped = np.reshape(arr, (np.prod(arr.shape[:])))
    return reshaped


def reshape_and_scale(X, y, scale, v_train_data, verbose=False):
    '''
    Reshapes and optionally scales the training data
    Scaling is performed manually and mins/maxs are saved
    for use in deployment.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)
    start_min, start_max = X_train.min(), X_train.max()

    if scale:
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
    
    # #  TODO: make this a function - test below
    # X_train_ss = np.reshape(X_train, (np.prod(X_train.shape[:-1]), X_train.shape[-1]))
    # X_test_ss = np.reshape(X_test, (np.prod(X_test.shape[:-1]), X_test.shape[-1]))
    # y_train = np.reshape(y_train, (np.prod(y_train.shape[:])))
    # y_test = np.reshape(y_test, (np.prod(y_test.shape[:])))

    X_train_ss = reshape_arr(X_train)
    X_test_ss = reshape_arr(X_test)
    y_train = reshape_arr(y_train)
    y_test = reshape_arr(y_test)

    if verbose:
        print(f'Reshaped X_train: {X_train_ss.shape} X_test: {X_test_ss.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}')
        print(f"The data was scaled to: Min {start_min} -> {end_min}, Max {start_max} -> {end_max}")
        
    return X_train_ss, X_test_ss, y_train, y_test