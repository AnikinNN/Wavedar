import datetime
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from . batch import Batch

class Logger:
    experiment = None

    def __init__(self, base_log_dir=None):
        if base_log_dir is None:
            base_log_dir = os.path.join(os.path.dirname(sys.argv[0]), 'logs')
        self.base_log_dir = base_log_dir
        make_dir(self.base_log_dir)

        self.experiment_number = self.get_experiment_number()
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tb_dir = os.path.join(self.base_log_dir, f'tb_{datetime_str}_{self.experiment_number}')
        self.misc_dir = os.path.join(self.base_log_dir, f'misc_{datetime_str}_{self.experiment_number}')

        for i in [self.tb_dir, self.misc_dir]:
            make_dir(i)

        self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

    def get_tb_writer(self):
        return self.tb_writer

    def get_experiment_number(self):
        numbers = set()
        for directory in os.listdir(self.base_log_dir):
            if re.match(r'(tb|misc)_\d{8}_\d{6}_\d+$', directory):
                numbers.add(int(directory.split('_')[-1]))
        return max(numbers) + 1 if len(numbers) else 1

    def store_batch_as_image(self, tag, batch: Batch, global_step=None,):
        waves = batch.significant_wave_height
        imgs = batch.images.detach().cpu().numpy()
        fig = plt.figure(figsize=(12, 12))
        fig.set_tight_layout(tight={'pad': -0.1, })
        square_size = np.ceil(np.sqrt(imgs.shape[0])).astype(int)

        for i in range(imgs.shape[0]):
            ax = fig.add_subplot(square_size, square_size, i + 1)
            im = ax.imshow(imgs[i, 0, :, :], cmap='Blues_r',
                           # origin='lower'
                           )
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.text(min(ax.get_xlim()), min(ax.get_ylim()),
                    f'{waves[i].item():.3f}',
                    horizontalalignment='left',
                    verticalalignment='top',
                    bbox={'pad': 0, 'color': 'white'}
                    )
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0)

        self.tb_writer.add_figure(tag, [fig], global_step)

    def store_scatter_hard_mining_weights(self, hard_mining_frame, epoch):
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot()
        x = hard_mining_frame['CM3up[W/m2]'].to_numpy()
        y = hard_mining_frame['hard_mining_weight'].to_numpy()
        ax.grid()
        ax.scatter(x, y, s=1)
        self.tb_writer.add_figure('hard_mining_weights', [fig], epoch)

    def store_target_vs_predicted(self, val_set, epoch):
        fig = plt.figure() # figsize=[8, 6])
        ax = fig.add_subplot()

        for cruise in val_set.wave_frame.cruise.unique():
            for station in val_set.wave_frame.station.unique():
                selection = (val_set.wave_frame.station == station) & (val_set.wave_frame.cruise == cruise)
                ax.scatter(val_set.wave_frame[selection].h, val_set.wave_frame[selection].last_predicted,
                           label=f'{cruise}_{station}',
                           alpha=0.5,
                           )

        ax.set_xlabel('true')
        ax.set_ylabel('predicted')
        ax.set_title('significant_wave_height, m')
        ax.grid()
        x_limit = [val_set.wave_frame.h.min(), val_set.wave_frame.h.max()]
        ax.plot(x_limit, x_limit, label='y=x')

        mean = [val_set.wave_frame.h.mean()]
        ax.plot(x_limit, [mean, mean], label='y=target_mean')

        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)

        self.tb_writer.add_figure('target_vs_predicted', [fig], epoch)


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
