import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .batch import Batch


class Resizer:
    def __init__(self):
        self.resizer = None

    def __call__(self, images, target_size):
        if self.resizer is None or self.resizer.size != target_size:
            self.init_resizer(target_size)

        return self.resizer(images)

    def init_resizer(self, target_size):
        self.resizer = torchvision.transforms.Resize(
            target_size, interpolation=InterpolationMode.BICUBIC)


class Sampler:
    def __init__(self):
        # affine numbers
        self.rotation_angle = self.sampler(10)
        self.scale = self.sampler(0.05, 1)
        self.translation_x = self.sampler(0.2)
        self.translation_y = self.sampler(0.2)

        # flip probabilities
        self.flip_ud = self.sampler(0.5, 0.5)
        self.flip_lr = self.sampler(0.5, 0.5)

        # gaussian noize additive
        self.noise_scale = self.sampler(0.01, 0.02)

    @staticmethod
    def sampler(delta: float = 1, center: float = 0):
        """get random float from uniform [center - delta, center + delta]"""
        return (np.random.rand() - 0.5) * 2 * delta + center


class Augmenter:
    normalizer = transforms.Normalize((0.456,), (0.224,))

    inv_normalizer = transforms.Compose([
        transforms.Normalize((0.,), (1 / 0.224,)),
        transforms.Normalize((-0.456,), (1.,)),
    ])

    resizer = Resizer()

    @classmethod
    def augment(cls, images: torch.Tensor, is_mask: bool, sampler: Sampler):
        if not is_mask:
            # add noise
            # add noize without normalization because it is followed by normalization
            noise_shape = [int(i * sampler.noise_scale) for i in images.shape[2:]]
            mean = torch.mean(images, dim=(2, 3))
            std = torch.std(images, dim=(2, 3))
            noise = cls.get_noise(mean, std * 0.1, noise_shape)
            additive = cls.resizer(noise, images.shape[-2:])
            images = images + additive

        min_image_side = min(images.shape[2:])
        images = transforms.functional.affine(images, sampler.rotation_angle,
                                              [sampler.translation_x * min_image_side,
                                               sampler.translation_y * min_image_side],
                                              sampler.scale,
                                              [0, 0],
                                              interpolation=transforms.InterpolationMode.NEAREST)

        if sampler.flip_ud > 0.5:
            images = torch.flip(images, dims=[1])
        if sampler.flip_lr > 0.5:
            images = torch.flip(images, dims=[2])

        if not is_mask:
            # normalize
            images = cls.normalizer(images)
        return images

    @classmethod
    def __call__(cls, batch: Batch):
        sampler = Sampler()
        images = cls.augment(batch.images, False, sampler)
        masks = cls.augment(batch.masks, True, sampler)
        return images * masks

    @classmethod
    def call(cls, batch: Batch):
        return cls.__call__(batch)

    @staticmethod
    def get_noise(mean: torch.Tensor, std: torch.Tensor, shape):
        assert mean.shape == std.shape
        noises = []
        for mean_i, std_i in zip(mean.reshape(-1), std.reshape(-1)):
            mean_i, std_i = mean_i.item(), std_i.item()
            noise_i = torch.normal(mean_i, std_i, size=(1, *shape)).to(mean.get_device())
            noises.append(noise_i)

        return torch.stack(noises, dim=0)
