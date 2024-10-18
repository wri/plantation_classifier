#! /usr/bin/env python3
import hickle as hkl
import yaml
import re
import pandas as pd
import numpy as np
import os
import utils.preprocessing as preprocess
import utils.validate_io as validate
import features.slow_glcm as slow_txt
from tqdm import tqdm
from skimage.util import img_as_ubyte
import json

def confirm_ard_alignment(v_train, local_dir, df_sample):
    '''
    TBD if needed --
    Cross references ARD generation survey w/ sample survey
    to ensure plot naming convention is executed properly
    
    Groups sample csv file and asserts there are 196
    rows per plot. Checks that each plot_fname in the
    plot csv exists in the sample csv
    '''
    df_plots = pd.read_csv(f'{local_dir}collect_earth_surveys/plantations-train-{v_train}/ceo-plantations-train-{v_train}-plot.csv')
    # plot_fnames = np.unique(df_plots.PLOT_FNAME)
    # sample_fnames = np.unique(df_sample.PLOT_FNAME)
    # assert (plot_fnames == sample_fnames).all(), print(f"Differences: {np.setxor1d(plot_fnames, sample_fnames)}")
    plot_counts = df_sample.groupby('PLOT_FNAME').size()
    for plot_fname, count in plot_counts.items():
        assert count == 196, f"Error: plot_fname {plot_fname} does not have 196 rows (has {count} rows)"
    print("All fname checks passed!")

def load_ceo_csv(v_train_data, local_dir):
    '''
    Cleans up the CEO sample survey in order to create label
    arrays by creating a plantation encoding. Ensures
    all input csvs have same format. 
    Creates the plot_fname using the same naming convention
    as ARD generation pipeline, except for the **SAMPLE** scale
    CEO survey.
    '''

    csv = f"{local_dir}ceo-plantations-train-{v_train_data}.csv"
    df = pd.read_csv(csv, encoding = "ISO-8859-1")

    df.columns = [re.sub(r'\W+', '', x) for x in df.columns]
    df.rename(columns={'ïplotid':'plotid'}, inplace=True)
    df.columns = [x.upper() for x in df.columns]
    df.columns = ['PLOT_ID' if x == 'PLOTID' else x for x in df.columns]
    df.columns = ['SAMPLE_ID' if x == 'SAMPLEID' else x for x in df.columns]

    df = df[['PLOT_ID', 
            'SAMPLE_ID', 
            'LON', 'LAT',
            'SYSTEM']]
    df['PLANTATION'] = df.SYSTEM.map({'Not plantation': 0, 
                                        'Monoculture': 1,
                                        'Agroforestry': 2,
                                        'Unknown': 255})
    
    # cross references naming convention in plot csv to ensure
    # alignment between ard files
    df['PLOT_FNAME'] = '0'
    plot_ids = []
    counter = 0
    for index, row in df.iterrows():
        if row['PLOT_ID'] not in plot_ids:
            plot_ids.append(row['PLOT_ID'])
            counter += 1
        
        df.loc[index, 'PLOT_FNAME'] = f"{str(v_train_data[1:]).zfill(2)}{str(counter).zfill(3)}"

    print("Writing clean csv...")
    df.to_csv(csv)

    confirm_ard_alignment(v_train_data, local_dir, df)

    return df


def reconstruct_images(plot, df):
    '''
    Takes a plot ID and subsets the input ceo survey (df) to that plot ID,
       computes the reverse of a given sequence object (lat) and 
       returns it in the form of a list.
       Requires presence of 'PLANTATION', 'LAT', 'LON' and 'PLOT_ID' columns 
       which should be created in prior processing step.
       Returns a (14, 14) array-like list with plantation labels.
    '''

    subs = df[df['PLOT_ID'] == plot]
    rows = []
    lats = reversed(sorted(subs['LAT'].unique()))

    for i, val in enumerate(lats):

        # filter to row
        subs_lat = subs[subs['LAT'] == val]
        subs_lat = subs_lat.sort_values('LON', axis = 0)
        rows.append(list(subs_lat['PLANTATION']))

    return rows

def create_label_arrays(v_train_data, local_dir):
    '''
    Checks if label arrays already exist on file.
    Assumption is that they don't need to be recreated.
    Iterates through each training data batch and 
    ensures csvs are in the correct format with 
    load_ceo_csv, then gathers the plot ids and
    creates the numpy label arrays. Saves to 
    output train-labels folder.
    '''
    directory = f"{local_dir}train-labels/"
    
    for i in v_train_data:
        labels_exist = any(file.startswith(i[1:]) for file in os.listdir(directory))
        if labels_exist:
            print(f"Label arrays exist for {i}, skipping")
        else:
            print(f"Creating label arrays for {i}")
            df = load_ceo_csv(i, local_dir)
            plot_ids = sorted(df['PLOT_ID'].unique())
            plot_fname = sorted(df['PLOT_FNAME'].unique())        
            for i, x in zip(plot_ids, plot_fname):
                #print(f"plot_ids:{i} plot_fname: {x}")
                plot = reconstruct_images(i, df)
                plot = np.array(plot)
                np.save(f"{directory}{str(x).zfill(5)}.npy", plot)
    
    return plot

def load_ard(idx, subsample, local_dir):
    """
    Analysis ready data is stored as (12, 28, 28, 13) with
    uint16 dtype, ranging from 0 - 65535 and ordered
    Sentinel-2, DEM, Sentinel-1.

    Converts to float32, removes border information and
    calculates median of full array or random subsample.

    (12, 28, 28, 13)
    (28, 28, 13)
    (14, 14, 13)
    """
    directory = f"{local_dir}train-ard/"
    ard = hkl.load(directory + str(idx) + ".hkl") 

    # checks for floating datatype, if not converts to float32
    if not isinstance(ard.flat[0], np.floating):
        assert np.max(ard) > 1
        ard = ard.astype(np.float32) / 65535
        assert np.max(ard) < 1

    # convert monthly images to subset median if subsample > 0
    # if no subset median on file, calculate and save
    if subsample > 0:
        if os.path.exists(f"{local_dir}train-ard-sub/{idx}.npy"):
            varied_median = np.load(f"{local_dir}train-ard-sub/{idx}.npy")
        else:
            rng = np.arange(12)
            indices = np.random.choice(rng, subsample, replace=False)
            varied_median = np.zeros((subsample, ard.shape[1], ard.shape[2], ard.shape[3]),
                                      dtype=np.float32)
            
            for x, i in zip(range(subsample), indices):
                varied_median[x, ...] = ard[i, ...]

            # calculate median w/ explicit setting to float32
            varied_median = np.median(np.float32(varied_median), 
                                        axis = 0, 
                                        overwrite_input = True)
            
            np.save(f"{local_dir}train-ard-sub/{idx}.npy", varied_median)

    else:
        varied_median = np.median(np.float32(ard), 
                                    axis=0, 
                                    overwrite_input = True)

    # slice out border information
    border_x = (varied_median.shape[0] - 14) // 2
    border_y = (varied_median.shape[1] - 14) // 2
    varied_median = varied_median[border_x:-border_x, border_y:-border_y, :]

    return varied_median


def load_txt(idx, local_dir):
    """
    S2 is stored as a (12, 28, 28, 11) uint16 array.

    Loads ARD data and filters to s2 indices. Preprocesses
    in order to extract texture features. Outputs the texture analysis
    as a (14, 14, 16) float32 array.

    Note that this will load the ARD median subset
    """
    directory = f"{local_dir}train-texture/"
    input_dir = f"{local_dir}train-ard-sub/"

    # check if texture file has already been created
    if os.path.exists(f"{directory}{idx}.npy"):
        output = np.load(f"{directory}{idx}.npy")
    else:
        ard = np.load(input_dir + str(idx) + ".npy")
        # if len(ard.shape) == 4:
        #     ard = np.median(ard, axis = 0, overwrite_input=True)
        s2 = ard[..., 0:10]
        s2 = img_as_ubyte(s2)
        # s2 = ((s2.astype(np.float32) / 65535) * 255).astype(np.uint8)
        assert s2.dtype == np.uint8, print(s2.dtype)
        blue = s2[..., 0]
        green = s2[..., 1]
        red = s2[..., 2]
        nir = s2[..., 3]
        output = np.zeros((14, 14, 16), dtype=np.float32)
        output[..., 0:4] = slow_txt.extract_texture(blue)
        output[..., 4:8] = slow_txt.extract_texture(green)
        output[..., 8:12] = slow_txt.extract_texture(red)
        output[..., 12:16] = slow_txt.extract_texture(nir)
        np.save(f"{directory}{idx}.npy", output)

    return output.astype(np.float32)


def load_ttc(idx, ttc_feats_dir, local_dir):
    """
    Features are stored as a 14 x 14 x 65 float64 array. The last axis contains
    the feature dimensions. Dtype needs to be converted to float32. The TML
    probability/prediction can optionally be dropped.

    ## Update per 2/13/23
    Features range from -infinity to +infinity
    and must be clipped to be consistent with the deployed features.

    Index 0 ([...,0]) is the tree cover prediction from the full TML model
    Index 1 - 33 are high level features
    Index 33 - 65 are low level features
    """

    directory = f"{local_dir}{ttc_feats_dir}"
    feats = hkl.load(directory + str(idx) + ".hkl")

    # clip all features after indx 0 to specific vals
    feats[..., 1:] = np.clip(feats[..., 1:], a_min=-32.768, a_max=32.767)

    feats = feats.astype(np.float32)

    return feats


def load_label(idx, ttc, classes, local_dir):
    """
    The labels are stored as a binary 14 x 14 float64 array.
    Unless they are stored as (196,) and need to be reshaped.
    Dtype needs to be converted to float32.

    For binary (2 class) classification, update labels by converting
    AF to 1. For 3 class classification, leave labels as is. For
    4 class classification, use the ttc data to update labels
    for any pixel labeled 0 with >= 20% TTC tree cover as natural trees.

    0: no tree
    1: monoculture
    2: agroforestry
    3: natural tree
    """
    directory = f"{local_dir}train-labels/"

    labels_raw = np.load(directory + str(idx) + ".npy")

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
        natree_mask = np.ma.masked_greater(tree_cover, 0.20000000)
        mask = np.logical_and(noplant_mask.mask, natree_mask.mask)
        labels[mask] = 3
        no_tree_mask = ttc[...,0] <= 0.1
        labels[no_tree_mask] = 0

    else:
        labels = labels_raw.astype(np.float32)

    return labels


def gather_plot_ids(v_train_data, local_dir, ttc_feats_dir, logger):
    """
    Cleans the downloaded CEO survey to meet requirements.
    Creates a list of plot ids to process. Drops all plots with
    "unknown" labels and plots w/o s2 imagery. Returns list of plot_ids.
    """

    # use CEO csv to gather plot id numbers
    plot_ids = []
    no_labels = []

    for i in v_train_data:
        df = pd.read_csv(f"{local_dir}ceo-plantations-train-{i}.csv")
        # assert unknown labels are always a full 14x14 (196 points) of unknowns
        unknowns = df[df.PLANTATION == 255]
        no_labels.extend(sorted(list(set(unknowns.PLOT_FNAME))))
        for plot in set(list(unknowns.PLOT_ID)):
            assert (
                len(unknowns[unknowns.PLOT_ID == plot]) == 196
            ), f"WARNING: {plot} has {len(unknowns[unknowns.PLOT_ID == plot])}/196 points labeled unknown."

        # drop unknowns and add to full list
        labeled = df.drop(unknowns.index)
        plot_ids += labeled.PLOT_FNAME.drop_duplicates().tolist()

    # add leading 0 to plot_ids that do not have 5 digits
    plot_ids = [str(item).zfill(5) if len(str(item)) < 5 else str(item) for item in plot_ids]
    final_ard = [plot for plot in plot_ids if os.path.exists(f"{local_dir}{ttc_feats_dir}{plot}.hkl")]
    no_ard = [plot for plot in plot_ids if not os.path.exists(f"{local_dir}{ttc_feats_dir}{plot}.hkl")]

    # remove problematic plots from cleanlab assessment
    with open("data/cleanlab/cleanlab_id_drops.json", "r") as file:
        cl_issues = json.load(file)
    cl_issues = set(cl_issues)
    final = [plot for plot in final_ard if plot not in cl_issues]

    logger.info(f'{len(no_labels)} plots labeled "unknown" were dropped.')
    logger.info(f"{len(no_ard)} plots did not have ARD.")
    logger.info(f"Training data batch includes: {len(final)} plots.")
    logger.info(f"Writing plot ids to file..")
    with open("data/cleanlab/plot_ids_v2.json", "w") as file:
        json.dump(final, file)

    return final


def make_sample(sample_shape, s2, slope, s1, txt, ttc):
    """
    Defines dimensions and then combines slope, s1, s2, TML features and
    texture features from a plot into a sample with shape (14, 14, 94)
    Feature select is a list of features that will be used, otherwise empty list
    Prepares sample plots by combining ARD and features
    and performing feature selection
    """
    # prepare the feats (this is done first bc of feature selection)
    # squeeze extra axis that is added (14,14,1,15) -> (14,14,15)
    feats = np.zeros(
        (sample_shape[0], sample_shape[1], ttc.shape[-1] + txt.shape[-1]),
        dtype=np.float32,
    )
    feats[..., : ttc.shape[-1]] = ttc
    feats[..., ttc.shape[-1] :] = txt

    # define the last dimension of the array
    n_feats = 1 + s1.shape[-1] + s2.shape[-1] + feats.shape[-1]
    sample = np.zeros((sample_shape[0], sample_shape[1], n_feats), dtype=np.float32)

    # populate empty array with each feature
    # order: s2, dem, s1, ttc, txt
    sample[..., 0:10] = s2
    sample[..., 10:11] = slope
    sample[..., 11:13] = s1
    sample[..., 13:] = feats

    return sample


def build_training_sample(train_batch, classes, params_path, logger):
    """
    Gathers training data plots from collect earth surveys (v1, v2, v3, etc)
    and loads data to create a sample for each plot. Removes ids where there is no
    cloud-free imagery or "unknown" labels.
    Create labels should be True any time a new CEO survey is added. This triggers
    csv cleaning and creates label arrays for the new training batch.

    Combines samples as X and loads labels as y for input to the model.
    """
    with open(params_path) as file:
        params = yaml.safe_load(file)

    train_data_dir = params["data_load"]["local_prefix"]
    ttc_feats_dir = params["data_load"]["ttc_feats_dir"]
    if params['data_load']['create_labels']:
        create_label_arrays(train_batch, train_data_dir)
    plot_ids = gather_plot_ids(train_batch, train_data_dir, ttc_feats_dir, logger)
    logger.info(f"{len(plot_ids)} plots will be used in training.")

    # create empty x and y array based on number of plots
    # x.shape is (plots, 14, 14, n_feats) y.shape is (plots, 14, 14)
    sample_shape = (14, 14)
    n_feats = params['data_condition']['total_feature_count']
    n_samples = len(plot_ids)
    y_all = np.zeros(shape=(n_samples, sample_shape[0], sample_shape[1]), dtype=np.float32)
    x_all = np.zeros(shape=(n_samples, sample_shape[0], sample_shape[1], n_feats), dtype=np.float32)
    med_indices = params["data_condition"]["ard_subsample"]

    for num, plot in enumerate(tqdm(plot_ids)):
        ard = load_ard(plot, med_indices, train_data_dir)
        ttc = load_ttc(plot, ttc_feats_dir, train_data_dir)
        txt = load_txt(plot, train_data_dir)
        validate.train_output_range_dtype(
            ard[..., 0:10],
            ard[..., 10:11],
            ard[..., 11:13],
            ttc,
        )
        X = make_sample(
            sample_shape,
            ard[..., 0:10],
            ard[..., 10:11],
            ard[..., 11:13],
            txt,
            ttc,
        )

        y = load_label(plot, ttc, classes, train_data_dir)
        x_all[num] = X
        y_all[num] = y
    
    # # save x and y
    print("Saving X and y on file")

   # Reshape the arrays to make them pixel-wise
    features_flat = x_all.reshape(-1, n_feats)  # Reshape to (total_pixels, n_feats)
    labels_flat = y_all.reshape(-1)  # Reshape to (total_pixels,)

    # Create a DataFrame where each row corresponds to a pixel
    df = pd.DataFrame(features_flat, columns=[f'feature_{i}' for i in range(n_feats)])
    df['label'] = labels_flat

    # Saving the DataFrame to a CSV file
    df.to_csv('data/cleanlab/cleanlab_demo3.csv', index=False)

    # check class balance
    labels, counts = np.unique(y_all, return_counts=True)
    class_dist = dict(zip(labels, counts))
    total = sum(class_dist.values())
    logger.info(f"Class count {class_dist}")
    for key, val in class_dist.items():
        class_dist[key] = round((val/total)*100,1)
    for key, val in class_dist.items():
        logger.info(f"{int(key)}: {val}%")

    return x_all, y_all
