import os.path
import glob
import re
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class MetadataLoader:
    def __init__(self, stations=None, split=(0.80, 0.10, 0.10), store_path=None):
        self.all_df = pd.DataFrame()
        self.train = pd.DataFrame()
        self.validation = pd.DataFrame()
        self.test = pd.DataFrame()
        self.npy_paths = []

        self.load_data(stations)

        self.all_df['buoy_datetime'] = pd.to_datetime(self.all_df['buoy_datetime'])
        self.all_df.sort_values(by="buoy_datetime", inplace=True)
        self.all_df['hard_mining_weight'] = 1.0
        self.all_df['npy_index'] = self.all_df['npy_index'].astype(int)
        self.all_df['last_predicted'] = np.nan
        self.split(*split)
        if store_path is not None:
            self.store_splits(store_path)

    @classmethod
    def init_using_data_dir(cls, data_dir, split=(0.80, 0.10, 0.10), store_path=None):
        stations = cls.find_dataset(data_dir)
        self = cls(stations, split, store_path)
        return self

    def load_data(self, stations):
        for csv, npy in stations:
            df = pd.read_csv(csv, sep=';')
            df['npy_path'] = npy
            self.extend_all(df)
            self.npy_paths.append(npy)

    @staticmethod
    def find_dataset(data_dir):
        files = []
        for file in os.listdir(data_dir):
            if re.match(r'\d+_target_meteo\.csv$', file):
                station = re.findall(r'\d+', file)[0]
                npy_path = os.path.join(data_dir, f'{station}_full_len.npy')
                if os.path.exists(npy_path):
                    files.append((os.path.join(data_dir, file), npy_path))
                else:
                    warnings.warn(f'found {file}, but not found {npy_path}')
        return files

    def extend_all(self, appendix):
        self.all_df = pd.concat((self.all_df, appendix), axis=0, ignore_index=True)

    def split(self, train_size, validation_size, test_size):
        assert 0.95 < (train_size + validation_size + test_size) <= 1, \
            'sum of train, validation, test must be less than 1'

        for cruise in self.all_df['cruise'].unique():
            for station in self.all_df['station'].unique():
                # bounds: [start, train_end, val_end, test_end]
                df = self.all_df[(self.all_df['station'] == station) & (self.all_df['cruise'] == cruise)]
                start = df['buoy_datetime'].min()
                end = df['buoy_datetime'].max()
                train_end = start + (end - start) * train_size
                val_end = start + (end - start) * (train_size + validation_size)

                self.train = pd.concat((self.train,
                                        df[(df['buoy_datetime'] > start) & (df['buoy_datetime'] <= train_end)]),
                                       axis=0, ignore_index=True)
                self.validation = pd.concat((self.validation,
                                             df[(df['buoy_datetime'] > train_end) & (df['buoy_datetime'] <= val_end)]),
                                            axis=0, ignore_index=True)
                self.test = pd.concat((self.test, df[df['buoy_datetime'] > val_end]), axis=0, ignore_index=True)

        for df, name in [(self.all_df, 'overall'),
                         (self.train, 'train'),
                         (self.validation, 'validation'),
                         (self.test, 'test')]:
            print(f'{name} len: {df.shape[0]}')

    def store_splits(self, path):
        for df, name in [(self.train, 'train'),
                         (self.validation, 'validation'),
                         (self.test, 'test')]:
            df.to_csv(os.path.join(path, f'subset_{name}.csv'))
