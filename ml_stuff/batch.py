from enum import Enum

import numpy as np
import torch
from torch.autograd import Variable


class BatchState(Enum):
    CPU_APPENDING = 1
    CPU_STORING = 2
    CUDA_STORING = 3


class Batch:
    def __init__(self):
        self.images = []
        self.masks = None
        self.significant_wave_height = []
        self.hard_mining_weights = []
        self.train_frame_indexes = []

        self.state = BatchState.CPU_APPENDING

    def __len__(self):
        return len(self.train_frame_indexes)

    def append(self, image: np.ndarray, significant_wave_height: float,
               hard_mining_weight: float, train_frame_index: int):
        if self.state is not BatchState.CPU_APPENDING:
            raise ValueError(f'You can append to batch only on FluxBatchState.CPU_APPENDING state. '
                             f'But there was an attempt on {self.state} state')

        self.images.append(image)
        self.significant_wave_height.append(significant_wave_height)
        self.hard_mining_weights.append(hard_mining_weight)
        self.train_frame_indexes.append(train_frame_index)

    def set_mask(self, mask: torch.Tensor):
        assert mask.shape[0] == self.__len__(), f'mask.shape[0] must be same as batch_size={self.__len__()} ' \
                                                f'but has value of{mask.shape[0]}'
        self.masks = mask

    def to_tensor(self):
        if self.state is not BatchState.CPU_APPENDING:
            raise ValueError(f'You can convert to tensor only from FluxBatchState.CPU_APPENDING state. '
                             f'But there was an attempt on {self.state} state')

        self.images = torch.stack(tuple(torch.tensor((i / 255.)[np.newaxis, :, :]) for i in self.images))
        # self.masks already tensors
        self.significant_wave_height = torch.reshape(torch.tensor(self.significant_wave_height), (-1, 1))
        self.hard_mining_weights = np.array(self.hard_mining_weights)

        self.state = BatchState.CPU_STORING

    def to_cuda(self, cuda_device, to_variable: bool):
        if self.state is not BatchState.CPU_STORING:
            raise ValueError(f'You can load to cuda device only on FluxBatchState.CPU_STORING state. '
                             f'But there was an attempt on {self.state} state')

        self.images = self.images.float()
        # self.masks = self.masks.float()
        self.significant_wave_height = self.significant_wave_height.float()

        if to_variable:
            self.images = Variable(self.images)
            # self.masks = Variable(self.masks)
            self.significant_wave_height = Variable(self.significant_wave_height)

        self.images = self.images.to(cuda_device)
        # self.masks = self.masks.to(cuda_device)
        self.significant_wave_height = self.significant_wave_height.to(cuda_device)

        self.state = BatchState.CUDA_STORING
