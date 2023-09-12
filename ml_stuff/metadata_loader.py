import os.path
import glob
import re
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class MetadataLoader:
    def __init__(self, stations=None, split=(0.80, 0.10, 0.10), logger=None, store_path=None, new_split=False,
                 use_slow_wind=True):
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
        self.all_df.wind_speed.fillna(self.all_df.wind_speed_airmar, inplace=True)
        self.all_df['last_predicted'] = np.nan
        self.save_h_frequency()
        self.logger = logger
        # self.split(*split)
        # self.split_by_stations(*split)
        slow_wind_threshold = 3
        low_h_threshold = 0.5
        if not use_slow_wind:
            self.all_df = self.all_df.drop(self.all_df[self.all_df['wind_speed'] < slow_wind_threshold].index)
            self.all_df = self.all_df.drop(self.all_df[self.all_df['h'] < low_h_threshold].index)
        self.stratified_split(*split, new_split)

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
            df.dropna(subset=['npy_index'], inplace=True)
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

    def save_h_frequency(self, bounds=np.arange(0, 5.1, 1)):
        counts, bins = np.histogram(self.all_df['h'].to_numpy(), bins=bounds)
        print(counts, 'counts')
        counts = counts / sum(counts)
        counts = 1 / counts
        index_list = np.clip(np.searchsorted(bins, self.all_df['h'].to_numpy()), 1, len(bounds)-1) - 1
        weights = pd.Series([counts[i] for i in index_list])
        self.all_df['weight'] = weights

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

    def stratified_split(self, train_size, validation_size, test_size, new_split):
        assert 0.95 < (train_size + validation_size + test_size) <= 1, \
            'sum of train, validation, test must be less than 1'
        station_means = []
        df_max = self.all_df['h'].max()
        for station in self.all_df['station'].unique():
            # bounds: [start, train_end, val_end, test_end]
            df = self.all_df[(self.all_df['station'] == station)]
            m = df['h'].mean()
            h_class = 5 - np.clip(int(df_max / m), 2, 5)
            station_means.append([station, h_class])
        station_means = np.array(station_means)
        if new_split:
            train, val_test = train_test_split(station_means, test_size=1-train_size, stratify=station_means[:, 1])
            val, test = train_test_split(val_test, test_size=test_size/(1-train_size), stratify=val_test[:, 1])
            train, val, test = train[:, 0], val[:, 0], test[:, 0]
            self.store_splits_ids('/app/wave/', train, val, test)
            print(len(train), len(val), len(test), 'len: train, val, test')
            if self.logger:
                self.store_splits_ids(self.logger.misc_dir, train, val, test)
        else:
            paths = [os.path.join('/app/wave/', i) for i in ['train.npy', 'val.npy', 'test.npy']]
            print(paths)
            train, val, test = self.load_split_ids(*paths)
        self.train = self.all_df[self.all_df['station'].isin(train)]
        self.validation = self.all_df[self.all_df['station'].isin(val)]
        self.test = self.all_df[self.all_df['station'].isin(test)]

        for df, name in [(self.all_df, 'overall'),
                         (self.train, 'train'),
                         (self.validation, 'validation'),
                         (self.test, 'test')]:
            print(f'{name} len: {df.shape[0]}')

    def split_by_stations(self, train_size, validation_size, test_size):
        stations_list = self.all_df['station'].unique()
        assert (train_size + validation_size + test_size) == len(stations_list), \
            'sum of train, validation, test must be equal to stations number'

        self.train = self.all_df[self.all_df['station'].isin(stations_list[:train_size])]
        self.validation = self.all_df[self.all_df['station'].isin(stations_list[train_size:train_size+validation_size])]
        self.test = self.all_df[self.all_df['station'].isin(stations_list[train_size+validation_size:])]

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

    def store_splits_ids(self, path, train, val, test):
        np.save(os.path.join(path, 'train.npy'), train)
        np.save(os.path.join(path, 'val.npy'), val)
        np.save(os.path.join(path, 'test.npy'), test)
    def load_split_ids(self, train_path, val_path, test_path):
        return np.load(train_path), np.load(val_path), np.load(test_path)

    def load_by_ids(self, train_path, val_path, test_path):
        train = np.load(train_path)
        val = np.load(val_path)
        test = np.load(val_path)
        self.train = self.all_df[self.all_df['station'].isin(train[:, 0])]
        self.validation = self.all_df[self.all_df['station'].isin(val[:, 0])]
        self.test = self.all_df[self.all_df['station'].isin(test[:, 0])]
        self.store_splits_ids(train, val, test)
        for df, name in [(self.all_df, 'overall'),
                         (self.train, 'train'),
                         (self.validation, 'validation'),
                         (self.test, 'test')]:
            print(f'{name} len: {df.shape[0]}')
