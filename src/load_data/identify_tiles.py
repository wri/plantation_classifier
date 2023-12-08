## import the CEO survey or database
import yaml
import pandas as pd
import os

def gather_plot_ids(v_train_data):
    '''
    Creates a list of plot ids to process from collect earth surveys 
    with multi-class labels (0, 1, 2, 255). Drops all plots with 
    "unknown" labels and plots w/o s2 imagery. Returns list of plot_ids.

    '''

    # use CEO csv to gather plot id numbers
    plot_ids = []
    no_labels = []

    for i in v_train_data:
        df = pd.read_csv(f'../data/ceo-plantations-train-{i}.csv')

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
    final_ard = [plot for plot in plot_ids if os.path.exists(f'../data/train-ard/{plot}.npy')]
    no_ard = [plot for plot in plot_ids if not os.path.exists(f'../data/train-ard/{plot}.npy')]
    final_raw = [plot for plot in no_ard if os.path.exists(f'../data/train-s2/{plot}.hkl')]

    print(f'{len(no_labels)} plots labeled "unknown" were dropped.')
    print(f'{len(no_ard)} plots did not have ARD; {len(final_raw)} plots will use raw data.')
    print(f'Training data batch includes: {len(final_ard)+len(final_raw)} plots.')

    return final_ard, final_raw
