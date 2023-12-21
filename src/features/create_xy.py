#! /usr/bin/env python3
import hickle as hkl
import yaml
import pandas as pd
import numpy as np
import os
import utils.preprocessing as preprocess
import utils.validate_io as validate
import features.slow_glcm as slow_txt
from tqdm import tqdm
from skimage.util import img_as_ubyte
from sklearn.model_selection import train_test_split

def load_slope(idx, local_dir):
    """
    Slope is stored as a 32 x 32 float32 array with border information.
    Needs to be converted to 1 x 14 x 14 array to match the labels w/ new axis.
    """
    directory = f'../{local_dir}train-slope/'

    # Remove axes of length one with .squeeze (32 x 32 x 1)
    x = np.load(directory + str(idx) + '.npy').squeeze()

    # slice out border information
    border_x = (x.shape[0] - 14) // 2
    border_y = (x.shape[1] - 14) // 2
    slope = x[border_x:-border_x, border_y:-border_y]

    # explicitly convert array to a row vector, adding axis (1 x 14 x 14)
    #slope = slope[np.newaxis]
    slope = slope[..., np.newaxis]
    
    return slope


def load_s1(idx, local_dir):
    """
    S1 is stored as a (12, 32, 32, 2) float64 array with border information.
    Needs to be converted from monthly mosaics to an annual median, 
    and remove border information to match labels. 
    Dtype needs to be converted to float32.
    """
    directory = f'../{local_dir}train-s1/'

    # since s1 is a float64, can't use to_float32() 
    s1 = hkl.load(directory + str(idx) + '.hkl')
    
    # convert to decible
    s1[..., -1] = preprocess.convert_to_db(s1[..., -1], 22)
    s1[..., -2] = preprocess.convert_to_db(s1[..., -2], 22)

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


def load_s2(idx, local_dir):
    
    """
    S2 is stored as a (12, 28, 28, 11) uint16 array. 
    Remove the last axis index - the date of the imagery. 
    Convert monthly images to an annual median. 
    Remove the border to correspond to the labels.
    Convert to float32.
    """
    directory = f'../{local_dir}train-s2/'
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


def load_ard(idx, subsample, local_dir):
    '''
    Analysis ready data is stored as (12, 28, 28, 13) with 
    uint16 dtype, ranging from 0 - 65535 and ordered 
    Sentinel-2, DEM, Sentinel-1.
    
    Converts to float32, removes border information and
    calculates median of full array or random subsample.

    (12, 28, 28, 13)
    (28, 28, 13)
    (14, 14, 13) 
    '''
    directory = f'../{local_dir}train-ard/'
    ard = np.load(directory + str(idx) + '.npy')

    # checks for floating datatype, if not converts to float32
    if not isinstance(ard.flat[0], np.floating):
        assert np.max(ard) > 1
        ard = ard.astype(np.float32) / 65535
        assert np.max(ard) < 1

    # convert monthly images to annual median
    if subsample > 0:
        rng = np.arange(12)
        indices = np.random.choice(rng, subsample, replace=False)
        varied_median = np.zeros((subsample, ard.shape[1], ard.shape[2], ard.shape[3]))

        for x in range(subsample):
            for i in indices:
                varied_median[x, ...] = ard[i, ...]

        med_ard = np.median(varied_median, axis = 0).astype(np.float32) # np.median changes dtype to float64
        np.save(f'../{local_dir}train-ard-sub/{idx}.npy', med_ard)
     
    else:
        med_ard = np.median(ard, axis = 0)

    # slice out border information
    border_x = (med_ard.shape[0] - 14) // 2
    border_y = (med_ard.shape[1] - 14) // 2
    med_ard = med_ard[border_x:-border_x, border_y:-border_y, :]
        
    return med_ard

def load_txt(idx, use_ard, local_dir):
    '''
    S2 is stored as a (12, 28, 28, 11) uint16 array. 
    
    Loads ARD data and filters to s2 indices. Preprocesses
    in order to extract texture features. Outputs the texture analysis 
    as a (14, 14, 16) float32 array.
    '''
    directory = f'../{local_dir}train-texture/'

    # ARD TEXTURE
    if use_ard:
        input_dir = f'../{local_dir}train-ard-sub/'
        # check if subset text has been created
        if os.path.exists(f'{directory}{idx}_sub.npy'):
            output = np.load(f'{directory}{idx}_sub.npy')
        else:
            ard = np.load(input_dir + str(idx) + '.npy')
            # if len(ard.shape) == 4:
            #     ard = np.median(ard, axis = 0, overwrite_input=True)
            s2 = ard[..., 0:10]
            s2 = img_as_ubyte(s2)
            #s2 = ((s2.astype(np.float32) / 65535) * 255).astype(np.uint8)  
            assert s2.dtype == np.uint8, print(s2.dtype)
            blue = s2[..., 0]
            green = s2[..., 1]
            red = s2[..., 2]
            nir = s2[..., 3]
            output = np.zeros((14, 14, 16))
            output[..., 0:4] = slow_txt.extract_texture(blue)
            output[..., 4:8] = slow_txt.extract_texture(green)
            output[..., 8:12] = slow_txt.extract_texture(red)
            output[..., 12:16] = slow_txt.extract_texture(nir)
            np.save(f'{directory}{idx}_sub.npy', output)
    
    # S2 TEXTURE
    else:
        input_dir = f'../{local_dir}train-s2/'
        if os.path.exists(f'{directory}{idx}.npy'):
            output = np.load(f'{directory}{idx}.npy')
        else: 
            s2 = hkl.load(input_dir + str(idx) + '.hkl')
            # remove date of imagery (last axis)
            # and convert monthly images to annual median
            if s2.shape[-1] == 11:
                s2 = np.delete(s2, -1, -1)
            if len(s2.shape) == 4:
                s2 = np.median(s2, axis = 0, overwrite_input=True)

            # this has to be done after median
            # doesnt work if just calling .astype(np.uint8)
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
            np.save(f'{directory}{idx}.npy', output)
    
    return output.astype(np.float32)

def load_ttc(idx, use_ard, local_dir):
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
    
    if use_ard:
        directory = f'../{local_dir}train-features-ard/'
    else:
        directory = f'../{local_dir}train-features-ckpt-2023-02-09/'

    feats = hkl.load(directory + str(idx) + '.hkl')

    # clip all features after indx 0 to specific vals
    feats[..., 1:] = np.clip(feats[..., 1:], a_min=-32.768, a_max=32.767)

    feats = feats.astype(np.float32)

    return feats


def load_label(idx, ttc, classes, local_dir):
    '''
    The labels are stored as a binary 14 x 14 float64 array.
    Unless they are stored as (196,) and need to be reshaped.
    Dtype needs to be converted to float32.
    
    For binary (2 class) classification, update labels by converting
    AF to 1. For 3 class classification, leave labels as is. For 
    4 class classification, use the ttc data to update labels
    for any vegetation >= 20% tree cover as natural trees.
    
    0: no tree
    1: monoculture
    2: agroforestry
    3: natural tree
    '''
    directory = f'../{local_dir}train-labels/'
    
    labels_raw = np.load(directory + str(idx) + '.npy')

    if len(labels_raw.shape) == 1:
        labels_raw = labels_raw.reshape(14, 14)

    if classes == 2:
        labels = labels_raw.copy()
        labels[labels_raw == 2] = 1
        labels = labels.astype(np.float32)

    if classes == 4:
        tree_cover = ttc[..., 0] 
        labels = labels_raw.copy()
        noplant_mask = np.ma.masked_less(labels, 1)
        notree_mask = np.ma.masked_greater(tree_cover, .20000000)
        mask = np.logical_and(noplant_mask.mask, notree_mask.mask)
        labels[mask] = 3
        
    else:
        labels = labels_raw.astype(np.float32)
    
    return labels

def gather_plot_ids(v_train_data, local_dir):
    '''
    Creates a list of plot ids to process from collect earth surveys 
    with multi-class labels (0, 1, 2, 255). Drops all plots with 
    "unknown" labels and plots w/o s2 imagery. Returns list of plot_ids.
    '''

    # use CEO csv to gather plot id numbers
    plot_ids = []
    no_labels = []

    for i in v_train_data:
        df = pd.read_csv(f'../{local_dir}ceo-plantations-train-{i}.csv')

        # assert unknown labels are always a full 14x14 (196 points) of unknowns
        unknowns = df[df.PLANTATION == 255]
        no_labels.extend(sorted(list(set(unknowns.PLOT_FNAME))))
        for plot in set(list(unknowns.PLOT_ID)):
            assert len(unknowns[unknowns.PLOT_ID == plot]) == 196,\
            f'WARNING: {plot} has {len(unknowns[unknowns.PLOT_ID == plot])}/196 points labeled unknown.'

        # drop unknowns and add to full list
        labeled = df.drop(unknowns.index)
        plot_ids += labeled.PLOT_FNAME.drop_duplicates().tolist()

    # add leading 0 to plot_ids that do not have 5 digits
    plot_ids = [str(item).zfill(5) if len(str(item)) < 5 else str(item) for item in plot_ids]
    final_ard = [plot for plot in plot_ids if os.path.exists(f'../{local_dir}train-ard/{plot}.npy')]
    no_ard = [plot for plot in plot_ids if not os.path.exists(f'../{local_dir}train-ard/{plot}.npy')]
    final_raw = [plot for plot in no_ard if os.path.exists(f'../{local_dir}train-s2/{plot}.hkl')]

    print(f'{len(no_labels)} plots labeled "unknown" were dropped.')
    print(f'{len(no_ard)} plots did not have ARD.')
    print(f'Training data batch includes: {len(final_ard)} plots.')

    return final_ard

def make_sample(sample_shape, s2, slope, s1, txt, ttc, feature_select):
    
    ''' 
    Defines dimensions and then combines slope, s1, s2, TML features and 
    texture features from a plot into a sample with shape (14, 14, 94)
    Feature select is a list of features that will be used, otherwise empty list
    Prepares sample plots by combining ARD and features 
    and performing feature selection
    '''
    # prepare the feats (this is done first bc of feature selection)
    # squeeze extra axis that is added (14,14,1,15) -> (14,14,15)
    feats = np.zeros((sample_shape[0], sample_shape[1], ttc.shape[-1] + txt.shape[-1]), dtype=np.float32)
    feats[..., :ttc.shape[-1]] = ttc
    feats[..., ttc.shape[-1]:] = txt
    if len(feature_select) > 0:
        feats = np.squeeze(feats[:, :, [feature_select]])

    # define the last dimension of the array
    n_feats = 1 + s1.shape[-1] + s2.shape[-1] + feats.shape[-1] 
    sample = np.empty((sample_shape[0], sample_shape[1], n_feats), dtype=np.float32)

    # populate empty array with each feature
    # order: s2, dem, s1, ttc, txt
    sample[..., 0:10] = s2
    sample[..., 10:11] = slope
    sample[..., 11:13] = s1
    sample[..., 13:] = feats

    return sample

def build_training_sample(v_train_data, classes, params_path, logger, feature_select=[]):
    '''
    Gathers training data plots from collect earth surveys (v1, v2, v3, etc)
    and loads data to create a sample for each plot. Removes ids where there is no
    cloud-free imagery or "unknown" labels.

    Combines samples as X and loads labels as y for input to the model. 
    Returns baseline accuracy score?

    TODO: finish documentation
    '''
    with open(params_path) as file:
        params = yaml.safe_load(file)
    
    train_data_dir = params['data_load']['local_prefix']
    plot_ids = gather_plot_ids(v_train_data, train_data_dir)
    
    if len(feature_select) > 0:
        n_feats = 13 + len(feature_select)
    else:
        n_feats = 94
        
    # create empty x and y array based on number of plots 
    # x.shape is (plots, 14, 14, n_feats) y.shape is (plots, 14, 14)
    sample_shape=(14, 14)
    n_samples = len(plot_ids)
    y_all = np.zeros(shape=(n_samples, sample_shape[0], sample_shape[1]))
    x_all = np.zeros(shape=(n_samples, sample_shape[0], sample_shape[1], n_feats))
    med_indices = params['data_condition']['ard_subsample']
    
    for num, plot in enumerate(tqdm(plot_ids)):
        ard = load_ard(plot, med_indices, train_data_dir)
        ttc = load_ttc(plot, True, train_data_dir)
        txt = load_txt(plot, True, train_data_dir) 
        validate.train_output_range_dtype(ard[...,0:10], 
                                          ard[...,10:11], 
                                          ard[...,11:13], 
                                          ttc, 
                                          feature_select) 
        X = make_sample(sample_shape, 
                        ard[...,0:10], 
                        ard[...,10:11], 
                        ard[...,11:13], 
                        txt, 
                        ttc, 
                        feature_select)
        y = load_label(plot, ttc, classes, train_data_dir)
        x_all[num] = X
        y_all[num] = y

        # clean up memory
        del ard, ttc, txt, X, y

    # for num, plot in enumerate(tqdm(raw)):
    #     slope = load_slope(plot, train_data_dir)
    #     s1 = load_s1(plot, train_data_dir)
    #     s2 = load_s2(plot, train_data_dir)
    #     ttc = load_ttc(plot, False, train_data_dir)
    #     txt = load_txt(plot, False, train_data_dir)
    #     validate.train_output_range_dtype(slope, s1, s2, ttc, feature_select)
    #     X = make_sample(sample_shape, s2, slope, s1, txt, ttc, feature_select)
    #     y = load_label(plot, ttc, classes, train_data_dir)
    #     x_all[num] = X
    #     y_all[num] = y
        
    #     del slope, s1, s2, ttc, txt, X, y
        
    # check class balance 
    labels, counts = np.unique(y_all, return_counts=True)
    #print(f'Class count {dict(zip(labels, counts))}')
    logger.info(f"Class count {dict(zip(labels, counts))}")

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


def reshape_and_scale(X, y, scale, v_train_data, params_path, logger):
    '''
    Reshapes x and y for input into a machine learning model. 
    Optionally scales the training data
    Scaling is performed manually and mins/maxs are saved
    for use in deployment.

    '''
    with open(params_path) as file:
        params = yaml.safe_load(file)

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=((params["data_condition"]["test_split"] / 100)),
                random_state=params["base"]["random_state"],
    )
    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
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
        np.save(f'../data/mins_{v_train_data}', min_all)
        np.save(f'../data/maxs_{v_train_data}', max_all)

    X_train_ss = reshape_arr(X_train)
    X_test_ss = reshape_arr(X_test)
    y_train = reshape_arr(y_train)
    y_test = reshape_arr(y_test)

    logger.info(
        f"Reshaped X_train: {X_train_ss.shape} X_test: {X_test_ss.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}"
    )
    logger.info(
        f"The data was scaled to: Min {start_min} -> {end_min}, Max {start_max} -> {end_max}"
    )

    return X_train_ss, X_test_ss, y_train, y_test