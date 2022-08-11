import threading
from queue import Queue

import numpy as np
import torch
from torch.autograd import Variable

from . batch_generator import WaveDataset
from . gpu_augmenter import Augmenter
from . positional_encoding import SinusoidalPositionalEmbedding


class ThreadKiller:
    """Boolean object for signaling a worker thread to terminate"""
    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_to_kill(self, to_kill):
        self.to_kill = to_kill


def threaded_batches_feeder(to_kill: ThreadKiller, target_queue: Queue, dataset_generator: WaveDataset):
    """
    takes batch from dataset_generator and put to target_queue until to_kill
    """
    for batch in dataset_generator:
        target_queue.put(batch, block=True)
        if to_kill():
            print('cpu_feeder_killed')
            return


def threaded_cuda_feeder(to_kill: ThreadKiller, target_queue: Queue, source_queue: Queue,
                         cuda_device, to_variable: bool, do_augment: bool):
    """
    takes batch from source_queue, transforms data to tensors, puts to target_queue until to_kill
    """
    while not to_kill():
        cuda_device = torch.device(cuda_device)
        batch = source_queue.get(block=True)
        batch.to_tensor()
        batch.to_cuda(cuda_device, to_variable)

        batch.images = ToCartesianConverter.__call__(batch.images)
        mask = WaveMask.get_mask(batch.images)
        assert mask is not None

        batch.set_mask(mask)

        if do_augment:
            batch.images, batch.masks, batch.significant_wave_height = \
                Augmenter.call(batch)

        batch.images = Augmenter.normalizer(batch.images)

        batch.images = batch.images * batch.masks
        target_queue.put(batch, block=True)
    print('cuda_feeder_killed')
    return


def threaded_cuda_augmenter(to_kill: ThreadKiller, target_queue: Queue, source_queue: Queue, do_augment: bool):
    """
    takes batch from source_queue, applies augmentations if do_augment, applies mask, puts to target_queue until to_kill
    """
    while not to_kill():
        batch = source_queue.get(block=True)
        if do_augment:
            batch.images = Augmenter.call(batch)
        target_queue.put(batch, block=True)
    print('cuda_feeder_killed')
    return


class ToCartesianConverter:
    xx_target, yy_target = np.meshgrid(np.linspace(0, 1, 480, dtype='float32'),
                                       np.linspace(-1, 1, 960, dtype='float32'))

    # avoid division by zero at tan calc
    tan = yy_target / np.where(xx_target == 0.0, 1, xx_target)
    theta_transform = np.arctan(tan) / np.pi * 2
    ro_transform = (np.hypot(yy_target, xx_target)) * 2 - 1

    grid = torch.tensor(np.stack([ro_transform, theta_transform], axis=-1)[np.newaxis, :, :, :])

    def __init__(self):
        pass

    @classmethod
    def __call__(cls, tensor_data: torch.Tensor):
        grid = cls.grid.expand(tensor_data.shape[0], -1, -1, -1).to(tensor_data.get_device()).float()
        res = torch.nn.functional.grid_sample(tensor_data, grid, padding_mode="zeros", mode='bilinear')
        return torch.rot90(res, 1, [2, 3])


class WaveMask:
    cached_mask = None
    step = 1.875  # m/step
    offset_meters = 300
    offset_steps = int(offset_meters // step)
    lock = threading.Lock()

    def __init__(self):
        pass

    @classmethod
    def get_mask(cls, images):
        with cls.lock:
            if cls.cached_mask is not None and images.shape == cls.cached_mask.shape:
                return cls.cached_mask
            else:
                polar_mask = np.ones(images.shape[2:])
                polar_mask[:, :cls.offset_steps] = 0

                tensor_polar_mask = torch.tensor(polar_mask[np.newaxis, np.newaxis, :, :]).to(images.get_device()).float()
                tensor_mask = ToCartesianConverter.__call__(tensor_polar_mask)
                tensor_mask = torch.where(tensor_mask > 0.5, 1.0, 0.0).expand(images.shape[0], -1, -1, -1)
                cls.cached_mask = tensor_mask
                print(f'Mask with shape {cls.cached_mask.shape} cached')
                return cls.cached_mask


class BatchFactory:
    def __init__(self,
                 dataset: WaveDataset,
                 cuda_device,
                 do_augment: bool,
                 cpu_queue_length: int = 4,
                 cuda_queue_length: int = 4,
                 preprocess_worker_number: int = 4,
                 cuda_feeder_number: int = 1,
                 to_variable: bool = True,
                 ):
        self.cpu_queue = Queue(maxsize=cpu_queue_length)
        self.cuda_queue = Queue(maxsize=cuda_queue_length)

        # one killer for all threads
        self.threads_killer = ThreadKiller()
        self.threads_killer.set_to_kill(False)

        # thread storage to watch after their closing
        self.cuda_feeders = []
        self.preprocess_workers = []

        for _ in range(cuda_feeder_number):
            thr = threading.Thread(target=threaded_cuda_feeder,
                                   args=(self.threads_killer,
                                         self.cuda_queue,
                                         self.cpu_queue,
                                         cuda_device,
                                         to_variable,
                                         do_augment)
                                   )
            thr.start()
            self.cuda_feeders.append(thr)

        for _ in range(preprocess_worker_number):
            thr = threading.Thread(target=threaded_batches_feeder,
                                   args=(self.threads_killer,
                                         self.cpu_queue,
                                         dataset))
            thr.start()
            self.preprocess_workers.append(thr)

    def stop(self):
        self.threads_killer.set_to_kill(True)

        # clean cuda_queues to stop cuda_feeder
        while sum(map(lambda x: int(x.is_alive()), self.cuda_feeders)):
            while not self.cuda_queue.empty():
                self.cuda_queue.get()

        # clean cpu_queues to stop preprocess_workers
        while sum(map(lambda x: int(x.is_alive()), self.preprocess_workers)):
            while not self.cpu_queue.empty():
                self.cpu_queue.get()
