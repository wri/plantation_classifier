import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rs
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import Point
from datetime import datetime
from sklearn.utils import resample
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn import metrics
import json


def confirm_ready(df):
    df = df.rename(columns={"System": "system"})
    missing_values = df[df['system'].isna()]
    if len(missing_values) > 0:
        print(f"WARNING: There are {len(missing_values)} samples with no label. Review the CEO survey.")
        return None
    df = df[['plotid', 'system', 'lon', 'lat', 'pl_pred']]    
    return df

def determine_label(row):
    '''
    If system_jess == system_john, the function immediately returns the matching value.
    If system_jess is 'unknown', return system_john.
    If system_john is 'unknown', return system_jess.
    If neither of the above conditions is met, return 'nc'.
    '''
    if row['system_jess'] == row['system_john']:
        return row['system_jess']
    
    return row['system_john'] if row['system_jess'] == 'Unknown' else (
           row['system_jess'] if row['system_john'] == 'Unknown' else 'nc')
    
def non_consensus_survey(df1, df2, version):
    '''
    df1 must be jess
    df2 must be john
    Creates a csv with non-consensus plots for 3rd party labeling
    Creates a clean csv with consensus plots for validation
    '''
    df1 = confirm_ready(df1)
    df2 = confirm_ready(df2)
    
    consensus = df1.merge(df2[['plotid', 'system']], 
                      on='plotid', 
                      suffixes=('_jess', '_john'))
    
    # create final label column
    consensus['final_label'] = consensus.apply(determine_label, axis=1)
    
    non_consensus = consensus[consensus.final_label == 'nc']
    print("Total non-consensus rows:", len(non_consensus))
    non_consensus = non_consensus[['plotid', 'lon', 'lat']]
    consensus = consensus[consensus.final_label != 'nc']

    non_consensus.to_csv(f'../../data/validation/non_consensus_labels_{version}.csv') ## Check this
    consensus.to_csv(f'../../data/validation/consensus_labels_{version}.csv')
    return consensus, non_consensus


def third_party_review(df1, df2, df3):
    '''
    df1 must be jess
    df2 must be john
    df3 must be daniel 
    Creates a csv that allocates daniel's labels to
    all plots where jess and john do not agree, based
    on the previous logic of defining consensus
    '''
    df1 = confirm_ready(df1)
    df2 = confirm_ready(df2)
    df3 = df3.rename(columns={"System": "system"})
    df3 = df3[['plotid', 'system', 'lon', 'lat']] 
    
    consensus = df1.merge(df2[['plotid', 'system', 'pl_pred']], 
                      on='plotid', 
                      suffixes=('_jess', '_john'))
    consensus['final_label'] = consensus.apply(determine_label, axis=1)

    # Create a mapping of plotid to system
    third_mapping = df3.set_index('plotid')['system']
    
    # incorporate 3rd party labels
    consensus.loc[consensus['final_label'] == 'nc', 'final_label'] = (
        consensus.loc[consensus['final_label'] == 'nc', 'plotid'].map(third_mapping)
    )
    assert 'nc' not in consensus['final_label'].values
    
    # clean up final output
    consensus = consensus[consensus.final_label != 'Unknown'].reset_index(drop=True)
    consensus = consensus.rename(columns={"final_label":"y_true",
                                         "pl_pred_jess": "y_pred"})
    pl_pred_map = {
        0: 'No vegetation',
        1: 'Monoculture',
        2: 'Agroforestry',
        3: 'Natural'}
    
    consensus['y_pred'] = consensus['y_pred'].map(pl_pred_map)
    consensus = consensus[['lon','lat','y_true', 'y_pred']]
    return consensus