import copy
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocessing import reshape_arr
from sklearn.utils.class_weight import compute_class_weight


class PlantationsData:
    def __init__(self, X_data_array, y_data_array, params):
        self.X_data_array = X_data_array
        self.y_data_array = y_data_array
        self.params = params

    def split_data(self):
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
        self.X_train_scaled = reshape_arr(X_train_tmp)
        self.X_test_scaled = reshape_arr(X_test_tmp)

    # TODO: save xmins and maxs

    def filter_features(self, feature_index_list):
        n_feats = len(feature_index_list)
        # create output arrays
        X_train_fs = np.zeros(
            shape=(
                self.X_train.shape[0],
                self.X_train.shape[1],
                self.X_train.shape[2],
                n_feats,
            )
        )
        X_test_fs = np.zeros(
            shape=(
                self.X_test.shape[0],
                self.X_test.shape[1],
                self.X_test.shape[2],
                n_feats,
            )
        )
        for i in range(self.X_train.shape[0]):
            X_train_fs[i : i + 1, ...] = self.X_train[
                i : i + 1, :, :, feature_index_list
            ]
        for i in range(self.X_train.shape[0]):
            X_test_fs[i : i + 1, ...] = self.X_test[i : i + 1, :, :, feature_index_list]

        self.X_train_reshaped = reshape_arr(X_train_fs)
        self.X_test_reshaped = reshape_arr(X_test_fs)
