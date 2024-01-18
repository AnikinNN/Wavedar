import warnings

import numpy as np
import os
import pandas as pd
import torch
from torch.nn import MSELoss

from tqdm import tqdm
from queue import Queue

from .batch_generator import WaveDataset
from .nn_logging import Logger
from .sgdr_restarts_warmup import CosineAnnealingWarmupRestarts
from .batch_factory import BatchFactory
from .gpu_augmenter import Augmenter
from .weighted_mse import WeightedMse, get_weights_and_bounds, BNILoss, BMCLoss, WeightedMSESavedWeights
from .resnet_regressor import ResnetRegressor


def train_single_epoch(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss_function,
                       cuda_batches_queue: Queue,
                       per_step_epoch: int,
                       current_epoch: int,
                       logger: Logger,
                       lr_scheduler
                       ):
    model.train()
    loss_values = []
    loss_tb = []
    pbar = tqdm(total=per_step_epoch, ncols=100)
    pbar.set_description(desc='train')

    warning_elapsed = False

    for batch_idx in range(per_step_epoch):
        batch = cuda_batches_queue.get(block=True)

        if batch_idx == 0:
            logger.store_batch_as_image('train_batch', batch,
                                        global_step=current_epoch)

        optimizer.zero_grad()
        data_out = model(batch.images)

        loss = loss_function(data_out, batch.significant_wave_height, batch.hard_mining_weights)
        loss_values.append(loss.item())

        loss_tb.append(loss.item())
        logger.tb_writer.add_scalar('train_loss_per_step', np.mean(loss_tb), current_epoch * per_step_epoch + batch_idx)
        loss_tb = []

        loss.backward()
        optimizer.step()
        pbar.update()
        pbar.set_postfix({'loss': loss.item(), 'cuda_queue_len': cuda_batches_queue.qsize()})

        if lr_scheduler is not None:
            lr_scheduler.step()
        else:
            if not warning_elapsed:
                warnings.warn('lr_scheduler is None')
                warning_elapsed = True

    pbar.close()

    return np.mean(loss_values)


def validate_single_epoch(model: torch.nn.Module,
                          loss_function,#: torch.nn.Module,
                          cuda_batches_queue: Queue,
                          per_step_epoch: int,
                          current_epoch: int,
                          logger: Logger,
                          val_dataset: WaveDataset
                          ):
    model.eval()
    loss_values = []
    loss_values1 = []

    pbar = tqdm(total=per_step_epoch, ncols=100)
    pbar.set_description(desc='validation')
    for batch_idx in range(per_step_epoch):
        batch = cuda_batches_queue.get(block=True)

        with torch.no_grad():
            data_out = model(batch.images)

        if batch_idx == 0:
            logger.store_batch_as_image('val_batch', batch,
                                        global_step=current_epoch)

        loss = loss_function[0](data_out, batch.significant_wave_height, batch.hard_mining_weights)
        loss1 = loss_function[1](data_out, batch.significant_wave_height)

        loss_values.append(loss.item())
        loss_values1.append(loss1.item())

        # save predicted to plot predicted(target)
        predicted = data_out.detach().cpu().numpy().squeeze()

        updater_df = pd.DataFrame({'last_predicted': predicted.tolist()})
        updater_df.set_index(pd.Index(batch.train_frame_indexes), inplace=True)
        val_dataset.wave_frame.update(updater_df)

        pbar.update()
        pbar.set_postfix({'loss': loss.item(), 'cuda_queue_len': cuda_batches_queue.qsize()})
    pbar.close()
    # plot predicted(target)
    logger.store_target_vs_predicted(val_dataset, current_epoch)

    return np.mean(loss_values), np.mean(loss_values1)


def calculate_hard_mining_weights(model: torch.nn.Module,
                                  train_dataset: WaveDataset,
                                  hard_mining_dataset: WaveDataset,
                                  cuda_batches_queue: Queue,
                                  logger: Logger,
                                  epoch: int
                                  ):
    model.eval()

    batch_number = len(hard_mining_dataset) // hard_mining_dataset.batch_size + 1
    pbar = tqdm(total=batch_number, ncols=100)
    pbar.set_description(desc='hard_mining')
    for batch_idx in range(batch_number):
        batch = cuda_batches_queue.get(block=True)

        with torch.no_grad():
            data_out = model(batch.images, batch.elevations)
            error = torch.abs(batch.fluxes - data_out).cpu().detach().numpy().flatten()

        error = error / error.mean()
        updater_df = pd.DataFrame({'hard_mining_weight': error.tolist()})
        updater_df.set_index(pd.Index(batch.train_frame_indexes), inplace=True)
        train_dataset.wave_frame.update(updater_df)

        pbar.update()
        pbar.set_postfix({'cuda_queue_len': cuda_batches_queue.qsize()})
    pbar.close()

    logger.store_scatter_hard_mining_weights(train_dataset.wave_frame, epoch)

    return


def train_model(model: ResnetRegressor,
                train_dataset: WaveDataset,
                val_dataset: WaveDataset,
                logger: Logger,
                cuda_device,
                learning_rate=5e-4,
                static_learning_rate=False,
                max_epochs=480,
                use_warmup: bool = False,
                steps_per_epoch_train=1536,
                steps_per_epoch_valid=1024,
                train_convolutional_since_epoch=None,

                ):
    mse = MSELoss()
    wmse = WeightedMSESavedWeights()
    # hist, bounds = np.histogram(train_dataset.wave_frame.h.to_numpy(), bins=100,)
    # bounds = (bounds[:-1] + bounds[1:]) / 2
    # mse = BMCLoss(1.0, cuda_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # lr_scheduler.step() calls every batch
    if static_learning_rate:
        lr_scheduler = None
    else:
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                     first_cycle_steps=14 * steps_per_epoch_train,  # 128*...?
                                                     cycle_mult=1.5,
                                                     max_lr=1e-4,
                                                     min_lr=5e-7,
                                                     warmup_steps=steps_per_epoch_train if use_warmup else 0,
                                                     gamma=0.8,
                                                     last_epoch=-1)

    train_batch_factory = BatchFactory(train_dataset,
                                       cuda_device,
                                       do_augment=True,
                                       preprocess_worker_number=4,
                                       to_variable=True,
                                       cuda_queue_length=2
                                       )
    validation_batch_factory = BatchFactory(val_dataset,
                                            cuda_device,
                                            do_augment=False,
                                            preprocess_worker_number=4,
                                            to_variable=False,
                                            cuda_queue_length=2
                                            )

    best_val_loss = float('Inf')
    best_val_epoch = -1
    best_val_losses = []
    best_val_epoches = []

    try:
        for epoch in range(max_epochs):
            print(f'Epoch {epoch} / {max_epochs}')

            # enable training of convolutional part at desired epoch
            if train_convolutional_since_epoch is None and train_convolutional_since_epoch == epoch:
                model.set_train_convolutional_part(True)

            train_loss = train_single_epoch(model,
                                            optimizer,
                                            wmse,
                                            train_batch_factory.cuda_queue,
                                            steps_per_epoch_train,
                                            current_epoch=epoch,
                                            logger=logger,
                                            lr_scheduler=lr_scheduler)
            logger.tb_writer.add_scalar('train_loss_per_epoch', train_loss, epoch)
            with open(os.path.join(logger.misc_dir, 'train.txt'), 'a') as f:
                f.write(str(epoch) + ' ' + str(train_loss) + '\n')

            val_loss, val_loss1 = validate_single_epoch(model,
                                             [wmse, mse],
                                             validation_batch_factory.cuda_queue,
                                             steps_per_epoch_valid,
                                             current_epoch=epoch,
                                             logger=logger,
                                             val_dataset=val_dataset)
            logger.tb_writer.add_scalar('val_loss', val_loss, epoch)
            with open(os.path.join(logger.misc_dir, 'val.txt'), 'a') as f:
                f.write(str(epoch) + ' ' + str(val_loss) + ' ' + str(val_loss1) + '\n')
            print(f'Validation loss: {val_loss}')

            if best_val_loss > val_loss:
                # save new model
                torch.save(model, os.path.join(logger.misc_dir, f'model_ep{epoch}.pt'))
                # delete old model
                old_model_path = os.path.join(logger.misc_dir, f'model_ep{best_val_epoch}.pt')
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
                best_val_loss = val_loss
                best_val_epoch = epoch
            if epoch == 13 or epoch == 33 or epoch == max_epochs-1:
                torch.save(model, os.path.join(logger.misc_dir, f'model_ep{epoch}.pt'))

    except KeyboardInterrupt:
        pass

    finally:
        for i in (
                train_batch_factory,
                validation_batch_factory,):
            i.stop()

    print('train_model: done')
    return best_val_loss
