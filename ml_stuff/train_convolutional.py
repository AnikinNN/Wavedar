import os
import sys

sys.path.append(os.path.join(sys.path[0], '..'))

import torch
import pandas as pd

from ml_stuff.batch_generator import WaveDataset
from ml_stuff.nn_logging import Logger
from ml_stuff.train_common import train_model
from ml_stuff.resnet_regressor import ResnetRegressor
from ml_stuff.metadata_loader import MetadataLoader

logger = Logger()

cuda_device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

batch_size = 16
encoder_dimension = 2

asv50_stations = (2763, 2777, 2792, 2809, 2821, 2833, 2841)
stations = tuple((f'/storage/tartar/suslovai/input_nn/input_nn_ASV50/target_ASV50/{i}_target_meteo.csv',
                  f'/storage/tartar/suslovai/input_nn/input_nn_ASV50/radar_data_ASV50/{i}_full_len.npy')
                 for i in asv50_stations)

metadata_loader = MetadataLoader(stations=stations, split=(0.7, 0.15, 0.15))

# # move 2792 2777 to validation
# selection = metadata_loader.train.station.isin([2792, 2777])
# metadata_loader.validation = pd.concat((metadata_loader.validation, metadata_loader.train[selection]),
#                                        ignore_index=True)
#
# metadata_loader.train = metadata_loader.train[~selection]

train_set = WaveDataset(wave_frame=metadata_loader.train,
                        batch_size=batch_size,
                        do_shuffle=True)

val_set = WaveDataset(wave_frame=metadata_loader.validation,
                      batch_size=batch_size,
                      do_shuffle=True)

modified_resnet = ResnetRegressor(encoder_dimension=encoder_dimension, first_conv_out=64)
modified_resnet.set_train_convolutional_part(True)
modified_resnet.to(cuda_device)

train_model(modified_resnet,
            train_dataset=train_set,
            val_dataset=val_set,
            logger=logger,
            cuda_device=cuda_device,
            max_epochs=64,
            use_warmup=True,
            steps_per_epoch_train=256,
            steps_per_epoch_valid=val_set.__len__() // batch_size + 1)

print()
