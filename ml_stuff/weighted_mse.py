import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def get_weights_and_bounds(target: np.ndarray):
    bounds = np.percentile(target, np.arange(0, 101))
    weights = bounds[1:] - bounds[:-1]
    # weights = np.histogram(target, bins=bounds, density=True)
    return weights / weights.mean(), bounds[1:-1]


class WeightedMse:
    def __init__(self, weights: np.ndarray, bounds: np.ndarray):
        for key, val in {'weights': weights, 'bounds': bounds}.items():
            if len(val.shape) > 1:
                warnings.warn(f'{key} has shape of {val.shape}. It will be flattened')

        weights = weights.flatten()
        bounds = bounds.flatten()

        assert weights.shape[0] - bounds.shape[0] == 1, 'weights must be 1 more than bounds'

        self.weights = weights
        self.bounds = bounds

    def __call__(self, predicted, target, hard_mining_weights):
        return self.weighted_mse_loss(predicted, target, hard_mining_weights)

    def weighted_mse_loss(self, predicted: torch.Tensor, target: torch.Tensor, hard_mining_weights: np.ndarray):
        target_ndarray = target.cpu().detach().numpy().flatten()
        weights_on_input = self.get_weights_on_input(target_ndarray)
        weights_on_input = torch.tensor(weights_on_input.reshape((-1, 1))).to(predicted.get_device())
        hard_mining_weights = torch.tensor(hard_mining_weights.reshape((-1, 1))).to(predicted.get_device())
        return (torch.square(predicted - target) * weights_on_input * hard_mining_weights).mean()

    def get_weights_on_input(self, target):
        weight_index = np.searchsorted(self.bounds, target, side='right')

        return self.weights[weight_index]


if __name__ == '__main__':
    # weights_ = np.arange(20, 25, 1)
    # bounds_ = np.arange(10, 14, 1)
    # target_ = np.arange(9, 15, 0.5)
    # loss_func = WeightedMse(weights_, bounds_)
    # print(target_, '\n', weights_, '\n', bounds_)
    # print(loss_func.get_weights_on_input(target_))

    target_ = np.random.rand(1000000) * 1200
    weights_, bounds_ = get_weights_and_bounds(target_)
    loss_func = WeightedMse(weights_, bounds_)


class BNILoss(_Loss):
    def __init__(self, init_noise_sigma, bucket_centers, bucket_weights, cuda_device):
        super(BNILoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device=cuda_device))
        self.bucket_centers = torch.tensor(bucket_centers, device=cuda_device)
        self.bucket_weights = torch.tensor(bucket_weights, device=cuda_device)

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bni_loss(pred, target, noise_var, self.bucket_centers, self.bucket_weights)
        return loss


def bni_loss(pred, target, noise_var, bucket_centers, bucket_weights):
    mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var

    num_bucket = bucket_centers.shape[0]
    bucket_center = bucket_centers.unsqueeze(0).repeat(pred.shape[0], 1)
    bucket_weights = bucket_weights.unsqueeze(0).repeat(pred.shape[0], 1)

    balancing_term = - 0.5 * (pred.expand(-1, num_bucket) - bucket_center).pow(2) / noise_var + bucket_weights.log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()
    return loss.mean()


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma, cuda_device):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device=cuda_device))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=pred.get_device()))
    loss = loss * (2 * noise_var).detach()

    return loss