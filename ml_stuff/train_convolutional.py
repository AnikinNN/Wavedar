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

cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 20
encoder_dimension = 2

asv50_stations = (2763, 2771, 2777, 2782, 2792, 2797, 2803, 2809, 2821, 2833, 2841,
                  2849, 2856, 2863, 2868, 2881, 2885, 2901, 2903, 2913, 2928, 2937,)
stations = tuple((f'/storage/tartar/suslovai/input_nn/input_nn_ASV50/target_ASV50/{i}_target_meteo.csv',
                  f'/storage/tartar/suslovai/input_nn/input_nn_ASV50/radar_data_ASV50/{i}_full_len.npy')
                 for i in asv50_stations)


def find_files(directory, pattern):
    import os, fnmatch
    flist = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\', '/')
                flist.append(filename)
    return flist


targets = []
radars = []
for cruise in ['AI57', 'AI58', 'AI63', 'ASV50']:
    fcsv = find_files(f'/storage/tartar/suslovai/input_nn/input_nn_{cruise}/target_{cruise}/10_min', '*.csv')
    fnpy = find_files(f'/storage/tartar/suslovai/input_nn/input_nn_{cruise}/radar_data_{cruise}/', '*.npy')
    print(len(fcsv), len(fnpy))
    targets = [*targets, *fcsv]
    radars = [*radars, *fnpy]

radars.sort()
targets.sort()
inputs = list(zip(targets, radars))
with open('stations.txt', 'a') as f:
    for item in inputs:
        f.write(str(item[0]) + ' ' + str(item[1]) + '\n')
    print('Done')
metadata_loader = MetadataLoader(stations=inputs, split=(0.7, 0.15, 0.15), logger=logger, new_split=True,
                                 use_slow_wind=True)
# metadata_loader = MetadataLoader(stations=inputs, split=(0.85, 0.05, 0.1))

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

modified_resnet = ResnetRegressor(encoder_dimension=encoder_dimension, first_conv_out=64, use_pos_encoding=True)
modified_resnet.set_train_convolutional_part(True)
modified_resnet.to(cuda_device)
print(modified_resnet)
with open(os.path.join(logger.misc_dir, 'description.txt'), 'a') as f:
    f.write('Нейросеть с Mish вместо relu, '
            'на новых данных с обновленным алгоритмом'
            ' поиска нужного сектора'
            'С позиционным кодированием со всеми данными' + '\n')
train_model(modified_resnet,
            train_dataset=train_set,
            val_dataset=val_set,
            logger=logger,
            cuda_device=cuda_device,
            max_epochs=64,
            use_warmup=True,
            steps_per_epoch_train=512,
            # steps_per_epoch_valid=val_set.__len__() // batch_size + 1)
            steps_per_epoch_valid=256)

print()
