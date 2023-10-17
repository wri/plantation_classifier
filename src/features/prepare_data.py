import hickle as hkl
import pandas as pd
import numpy as np
import os
import yaml
from tqdm import tqdm
from utils.logs import get_logger
import data.clean_ceo_summary as cc


def get_ceo_plot_ids(v_train_data, config_path, label_type=['binary']):
    '''
    Creates a list of plot ids to process from collect earth surveys 
    with binary class labels (0, 1). Drops all plots w/o s2 imagery. 
    Returns list of plot_ids.
    '''
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    # use CEO csv to gather plot id numbers
    plot_ids = []

    for i in v_train_data:
        ceo_path = config['data_load']['ceo_survey_directory']
        df = pd.read_csv(f'{ceo_path}/ceo-plantations-train-{i}.csv')
        ceo_summary = cc.import_ceo_summary(config_path)
        multiclass_batches =cc.get_batch_numbers(cc.ceo_filter(ceo_summary, 'Classes', 'multi'))
        if i in multiclass_batches:        
            # map label categories to ints
            # this step might not be needed but leaving for now.
            df['PLANTATION_MULTI'] = df['PLANTATION'].map({'Monoculture': 1,
                                                        'Agroforestry': 2,
                                                        'Not plantation': 0,
                                                        'Unknown': 255})

            # confirm that unknown labels are always a full 14x14 (196 points) of unknowns
            # if assertion fails, will print count of points
            unknowns = df[df.PLANTATION_MULTI == 255]
            for plot in set(list(unknowns.PLOT_ID)):
                assert len(unknowns[unknowns.PLOT_ID == plot]) == 196, f'{plot} has {len(unknowns[unknowns.PLOT_ID == plot])}/196 points labeled unknown.'
            # drop unknown samples
            df_new = df.drop(df[df.PLANTATION_MULTI == 255].index)
            print(f'{(len(df) - len(df_new)) / 196} plots labeled unknown were dropped from {i}.')
            
            plot_ids = plot_ids + df_new.PLOT_FNAME.drop_duplicates().tolist()
        else:
            # assert unknown labels are always a full 14x14 (196 points) of unknowns
            unknowns = df[df.PLANTATION == 255]
            for plot in set(list(unknowns.PLOT_ID)):
                assert len(unknowns[unknowns.PLOT_ID == plot]) == 196,\
                f'{plot} has {len(unknowns[unknowns.PLOT_ID == plot])}/196 points labeled unknown.'

            # drop unknowns and add to full list
            df_new = df.drop(df[df.PLANTATION == 255].index)
            print(f'{int((len(df) - len(df_new)) / 196)} plots labeled unknown were dropped from {i}.')
        plot_ids += df_new.PLOT_FNAME.drop_duplicates().tolist()
    # if the plot_ids do not have 5 digits, change to str and add leading 0
    plot_ids = [str(item).zfill(5) if len(str(item)) < 5 else str(item) for item in plot_ids]
    # check and remove any plot ids where there are no cloud free images (no s2 hkl file)
    print('warning needs to be updated')
    for plot in plot_ids[:]:   
        local_path= config['data_load']['local_prefix']         
        if not os.path.exists(f'{local_path}/train-s2/{plot}.hkl'.strip()):
            print(f'Plot id {plot} has no cloud free imagery and will be removed.')
            plot_ids.remove(plot)
    return plot_ids

def binary_ceo(v_train_data, config_path):
    '''
    Creates a list of plot ids to process from collect earth surveys 
    with binary class labels (0, 1). Drops all plots w/o s2 imagery. 
    Returns list of plot_ids.
    '''
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    # use CEO csv to gather plot id numbers
    plot_ids = []

    for i in v_train_data:
        ceo_path = config['data_load']['ceo_survey_directory']
        df = pd.read_csv(f'{ceo_path}/ceo-plantations-train-{i}.csv')
        
        # for multiclass surveys, change labels
        multiclass = ['v08', 'v14', 'v15']
        if i in multiclass:
        
            # map label categories to ints
            # this step might not be needed but leaving for now.
            df['PLANTATION_MULTI'] = df['PLANTATION'].map({'Monoculture': 1,
                                                        'Agroforestry': 2,
                                                        'Not plantation': 0,
                                                        'Unknown': 255})

            # confirm that unknown labels are always a full 14x14 (196 points) of unknowns
            # if assertion fails, will print count of points
            unknowns = df[df.PLANTATION_MULTI == 255]
            for plot in set(list(unknowns.PLOT_ID)):
                assert len(unknowns[unknowns.PLOT_ID == plot]) == 196, f'{plot} has {len(unknowns[unknowns.PLOT_ID == plot])}/196 points labeled unknown.'

            # drop unknown samples
            df_new = df.drop(df[df.PLANTATION_MULTI == 255].index)
            print(f'{(len(df) - len(df_new)) / 196} plots labeled unknown were dropped from {i}.')
            
            plot_ids = plot_ids + df_new.PLOT_FNAME.drop_duplicates().tolist()

        # for binary surveys add to list
        else:
            plot_ids = plot_ids + df.PLOT_FNAME.drop_duplicates().tolist()

    # if the plot_ids do not have 5 digits, change to str and add leading 0
    plot_ids = [str(item).zfill(5) if len(str(item)) < 5 else str(item) for item in plot_ids]

    # check and remove any plot ids where there are no cloud free images (no s2 hkl file)
    print('warning needs to be updated')
    for plot in plot_ids[:]:   
        local_path= config['data_load']['local_prefix']         
        if not os.path.exists(f'{local_path}/train-s2/{plot}.hkl'.strip()):
            print(f'Plot id {plot} has no cloud free imagery and will be removed.')
            plot_ids.remove(plot)
    return plot_ids

def multiclass_ceo(v_train_data, config_path):
    '''
    Creates a list of plot ids to process from collect earth surveys 
    with multi-class labels (0, 1, 2, 255). Drops all plots with 
    "unknown" labels and plots w/o s2 imagery. Returns list of plot_ids.
    '''
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    # use CEO csv to gather plot id numbers
    plot_ids = []

    for i in v_train_data:
        ceo_path = config['data_load']['ceo_survey_directory']
        # for each training data survey, drop all unknown labels
        df = pd.read_csv(f'{ceo_path}/ceo-plantations-train-{i}.csv')

        # assert unknown labels are always a full 14x14 (196 points) of unknowns
        unknowns = df[df.PLANTATION == 255]
        for plot in set(list(unknowns.PLOT_ID)):
            assert len(unknowns[unknowns.PLOT_ID == plot]) == 196,\
            f'{plot} has {len(unknowns[unknowns.PLOT_ID == plot])}/196 points labeled unknown.'

        # drop unknowns and add to full list
        df_new = df.drop(df[df.PLANTATION == 255].index)
        print(f'{int((len(df) - len(df_new)) / 196)} plots labeled unknown were dropped from {i}.')
        plot_ids += df_new.PLOT_FNAME.drop_duplicates().tolist()

    # add leading 0 to plot_ids that do not have 5 digits
    plot_ids = [str(item).zfill(5) if len(str(item)) < 5 else str(item) for item in plot_ids]
        
    # remove any plot ids where there are no cloud free images (no s2 hkl file)
    local_path= config['data_load']['local_prefix']
    final_plots = [plot for plot in plot_ids if os.path.exists(f'{local_path}/train-s2/{plot}.hkl')]
    print(f'{len(plot_ids) - len(final_plots)} plots had no cloud free imagery and will be removed.')

    return final_plots

def create_xy(v_train_data, classes, drop_feats, config_path, feature_select=[], verbose=False):
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
    
    # need to be able to create xy for 1) binary only 2) multiclass only 3) binary and multi
    plot_ids = get_ceo_plot_ids(v_train_data, config_path, classes)
    
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
            y = load_label(plot, classes)
            x_all[num] = X
            y_all[num] = y

        else:
            slope = load_slope(plot)
            s1 = load_s1(plot)
            s2 = load_s2(plot)
            ttc = load_ttc(plot)
            txt = load_txt(plot)
            #print('REMINDER: Import fast or slow txt feats?')
            #feats = load_feats(plot, import_txt=False)
            validate.train_output_range_dtype(slope, s1, s2, ttc, feature_select)
            X = make_sample(sample_shape, slope, s1, s2, txt, ttc, feature_select)
            y = load_label(plot, classes)
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