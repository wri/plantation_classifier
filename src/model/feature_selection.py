#!/usr/bin/env python

import pickle
import pandas as pd
from catboost import CatBoostClassifier


def feature_selection(model, feat_count):
    
    '''
    Calculates the feature importance score for a given model
    using CatBoost's default feature importance score - FeatureImportance
    Returns df containing importance score for each feature and
    top n (feat_count) most important features
    '''
    
    filename = f'../models/{model}.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    # calculate the feature importance 
    df = model.get_feature_importance(prettified=True)
    df = df.astype({'Feature Id': int})

    # filter to features > index 13 (remove importance for s1, s2, dem)
    # get the indices and print in ascending order
    feats_df = df[df['Feature Id'] >= 13]
    top_feats = feats_df.sort_values(by='Importances', ascending=False)[:feat_count]
    top_feats_indices = [i - 13 for i in sorted(list(top_feats['Feature Id']))]

    return df, top_feats_indices