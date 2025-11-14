import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


class StandardScaler:
    """
    Minimal replacement for sklearn.preprocessing.StandardScaler.
    Avoids heavy SciPy/Sklearn dependency while providing the few
    methods this project needs.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, data):
        arr = np.asarray(data, dtype=np.float64)
        self.mean_ = np.nanmean(arr, axis=0)
        self.scale_ = np.nanstd(arr, axis=0)
        # Replace fully-missing columns with safe defaults
        self.mean_ = np.nan_to_num(self.mean_, nan=0.0)
        self.scale_ = np.nan_to_num(self.scale_, nan=1.0)
        self.scale_[self.scale_ == 0] = 1.0  # prevent divide-by-zero
        return self

    def transform(self, data):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before calling transform.")
        arr = np.asarray(data, dtype=np.float64)
        arr = np.where(np.isnan(arr), self.mean_, arr)
        scaled = (arr - self.mean_) / self.scale_
        return np.nan_to_num(scaled, nan=0.0)

    def inverse_transform(self, data):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before calling inverse_transform.")
        arr = np.asarray(data, dtype=np.float64)
        return np.nan_to_num(arr * self.scale_ + self.mean_, nan=0.0)


def _clean_numeric_columns(df, feature_cols):
    """
    Replace NaNs/Infs before scaling so the model never sees undefined values.
    """
    if not len(feature_cols):
        return df
    numeric = df.loc[:, feature_cols].apply(pd.to_numeric, errors='coerce')
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.interpolate(method='linear', limit_direction='both', axis=0)
    col_means = numeric.mean(axis=0)
    numeric = numeric.fillna(col_means)
    numeric = numeric.fillna(0.0)
    df = df.copy()
    df.loc[:, feature_cols] = numeric
    return df

drop_num = 10

class Dataset_RCA(Dataset):
    def __init__(self, root_path, flag, 
                 data_path, target, 
                 features='MS', size=None, scale=True, timeenc=0, freq='s', train_only=False):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test']
        type_map = {'train': 0,'test': 1}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        cols_data = df_raw.columns[1:]
        df_raw = _clean_numeric_columns(df_raw, cols_data)

        data_len = len(df_raw)

        border1s = [0, int(data_len*(3/4)) - self.seq_len]
        border2s = [int(data_len*(3/4)), data_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[cols_data]

        train_data = df_data[border1:border2]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        clean_slice = np.nan_to_num(data, nan=0.0)
        self.data_x = clean_slice[border1:border2]
        self.data_y = clean_slice[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y
        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
