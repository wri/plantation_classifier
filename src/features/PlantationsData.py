import copy
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocessing import reshape_arr
from sklearn.utils.class_weight import compute_class_weight
import os
import json

class PlantationsData:
    '''
    For the data split, if use_precomputed_split == True, imports list of train 
    and val ids that were created in build_training_sample().
    Uses the plot_fname to map plot IDs back to array row positions, to 
    guarantee the exact same plots are used to build train_ids.txt and val_ids.txt
    '''
    def __init__(self, X_data_array, y_data_array, params):
        self.X_data_array = X_data_array
        self.y_data_array = y_data_array
        self.params = params

    def split_data(self):
        use_precomputed_split = self.params["train"]["use_split"]
        split_path = self.params["data_load"]["local_prefix"] + 'train_params/'
        train_file = os.path.join(split_path, "train_ids.txt")
        val_file = os.path.join(split_path, "val_ids.txt")
        all_ids_file = os.path.join(self.params["data_load"]["local_prefix"], "cleanlab/round2/final_plot_ids.json")

        if use_precomputed_split:
            with open(train_file) as f:
                train_ids = [int(x.strip()) for x in f]
            with open(val_file) as f:
                val_ids = [int(x.strip()) for x in f]
            with open(all_ids_file) as f:
                all_ids = json.load(f)

            # Map PLOT_FNAME to index
            id_to_idx = {pid: idx for idx, pid in enumerate(all_ids)}
            train_idx = [id_to_idx[i] for i in train_ids]
            val_idx = [id_to_idx[i] for i in val_ids]

            self.X_train = self.X_data_array[train_idx]
            self.X_test = self.X_data_array[val_idx]
            self.y_train = self.y_data_array[train_idx]
            self.y_test = self.y_data_array[val_idx]
        else:
            # fallback to default
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_data_array,
                self.y_data_array,
                test_size=((self.params["data_condition"]["test_split"] / 100)),
                train_size=((self.params["data_condition"]["train_split"] / 100)),
                random_state=self.params["base"]["random_state"],
            )


    def reshape_data_arr(self):
        self.X_train_reshaped = reshape_arr(self.X_train)
        self.X_test_reshaped = reshape_arr(self.X_test)
        self.y_train_reshaped = (reshape_arr(self.y_train)).astype(int)
        self.y_test_reshaped = (reshape_arr(self.y_test)).astype(int)
        self.class_names = np.unique(self.y_train_reshaped)
        self.class_weights = compute_class_weight(
            class_weight="balanced", classes=self.class_names, y=self.y_train_reshaped
        )

    def scale_X_arrays(self):
        """
        Performs manual scaling of training data
        """
        X_train_tmp = copy.deepcopy(self.X_train)
        X_test_tmp = copy.deepcopy(self.X_test)

        # standardize train/test data 
        min_all = []
        max_all = []
        
        for band in range(0, self.X_train.shape[-1]):
            mins = np.percentile(self.X_train[..., band], 1)
            maxs = np.percentile(self.X_train[..., band], 99)

            if maxs > mins:
                # clip values in each band based on min/max of training dataset
                X_train_tmp[..., band] = np.clip(X_train_tmp[..., band], mins, maxs)
                X_test_tmp[..., band] = np.clip(X_test_tmp[..., band], mins, maxs)

                # calculate standardized data
                midrange = (maxs + mins) / 2
                rng = maxs - mins
                X_train_tmp[..., band] = (X_train_tmp[..., band] - midrange) / (rng / 2)
                X_test_tmp[..., band] = (X_test_tmp[..., band] - midrange) / (rng / 2)
                min_all.append(mins)
                max_all.append(maxs)
                
        self.mins = min_all
        self.maxs = max_all
        self.X_train_scaled = reshape_arr(X_train_tmp)
        self.X_test_scaled = reshape_arr(X_test_tmp)

    def filter_features(self, feature_index_list):
        '''
        Filters X_train and X_test with shape (x, 14, 14, 94)
        to identified set of features and reshapes the array 
        into 2 dimensions for input into model.
        '''

        X_train_reshaped = np.squeeze(self.X_train[:, :, :, [feature_index_list]])
        X_test_reshaped = np.squeeze(self.X_test[:, :, :, [feature_index_list]])
    
        self.X_train_reshaped = reshape_arr(X_train_reshaped)
        self.X_test_reshaped = reshape_arr(X_test_reshaped)
