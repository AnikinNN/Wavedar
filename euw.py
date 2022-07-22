import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import os, sys

sys.path.append(os.path.join(os.getcwd(), '../..'))

from ml_stuff.metadata_loader import MetadataLoader
from ml_stuff.batch_generator import WaveDataset
from ml_stuff.batch_factory import BatchFactory

from tqdm import tqdm, trange

plt.rcParams['figure.dpi'] = 200

loader = MetadataLoader((('./fake_csv.csv', './fake_npy.npy'), ))
dataset = WaveDataset(loader.test)

factory = BatchFactory(dataset=dataset,
                       cuda_device='cuda:1',
                       do_augment=True,
                       )

factory.cpu_queue.qsize()

for _ in trange(100):
    batch = factory.cuda_queue.get()

factory.stop()
