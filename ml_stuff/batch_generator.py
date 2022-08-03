import threading
import numpy as np

from sklearn.utils import shuffle

from . batch import Batch
from . threadsafe_iterator import ThreadsafeIterator


def get_object_index(objects_count):
    """Cyclic generator of indices from 0 to objects_count
    """
    current_id = 0
    while True:
        yield current_id
        current_id = (current_id + 1) % objects_count


class WaveDataset:
    def __init__(self, wave_frame, batch_size=32, do_shuffle=True):
        self.wave_frame = wave_frame
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle
        self.shuffle_data()

        self.objects_iloc_generator = ThreadsafeIterator(get_object_index(self.wave_frame.shape[0]))
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch

        self.batch = Batch()
        self.npys = {}

        self.significant_wave_height_mean = wave_frame.h.mean()
        self.significant_wave_height_std = wave_frame.h.std()

    def __len__(self):
        return self.wave_frame.shape[0]

    def shuffle_data(self):
        if self.do_shuffle:
            self.wave_frame = shuffle(self.wave_frame)

    def get_data_by_id(self, index):
        image = self.get_image(index)
        significant_wave_height = self.wave_frame.iloc[index]['h']
        row_id = self.wave_frame.iloc[index].name
        hard_mining_weight = self.wave_frame.iloc[index]['hard_mining_weight']

        return image, significant_wave_height, hard_mining_weight, row_id

    def get_image(self, index):
        npy = self.get_npy(index)
        npy_index = self.wave_frame.iloc[index]['npy_index']
        image = np.array(npy[npy_index, :, :]).squeeze().astype('float')

        return image

    def get_npy(self, index):
        npy_path = self.wave_frame.iloc[index]['npy_path']
        if npy_path not in self.npys:
            npy = np.load(npy_path, mmap_mode='r')
            self.npys[npy_path] = npy
        return self.npys[npy_path]

    def __iter__(self):
        while True:
            for obj_iloc in self.objects_iloc_generator:
                image, significant_wave_height, hard_mining_weight, row_id = self.get_data_by_id(obj_iloc)

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if len(self.batch) < self.batch_size:
                        self.batch.append(image, significant_wave_height, hard_mining_weight, row_id)

                    if len(self.batch) >= self.batch_size:
                        yield self.batch
                        self.clean_batch()

    def clean_batch(self):
        self.batch = Batch()
