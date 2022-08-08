import threading

import torch


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, encoding_dimension):
        super().__init__()
        self.encoding_dimension = encoding_dimension
        self.omega = self.get_omega(self.encoding_dimension)
        self.cache = torch.Tensor()

    @staticmethod
    def get_omega(dimension):
        if dimension != 0:
            return 1 / (1e4 ** (torch.arange(0, 1, 1 / dimension)))
        else:
            return torch.Tensor()

    def encode(self, batch: torch.Tensor):
        """
        returns
        :param batch: size (batch_size, channels, H, W, ...)
        :return: size[batch_size, 2(sin/cos) * 2(x/y) * dimension, H, W, ...]
        """
        data_shape = batch.shape[2:]
        encoding_shape = self.get_encoding_shape(batch)

        if self.cache.shape != encoding_shape:
            encoding = torch.zeros(encoding_shape, device=batch.get_device())
            for spatial_dimension_i, spatial_dimension_len in enumerate(data_shape):
                # x.shape == [spatial_dimension_len]
                x = torch.arange(spatial_dimension_len, device=batch.get_device()).float()

                # move omega to cuda
                self.omega = self.omega.to(batch.get_device())
                # angle.shape == [self.encoding_dimension, spatial_dimension_len]
                angle = torch.matmul(self.omega.unsqueeze(1), x.unsqueeze(0))

                # sin_x.shape == angle.shape
                sin_x = torch.sin(angle)
                cos_x = torch.cos(angle)

                # sin_xx.shape ==[self.encoding_dimension, H, W, ...]
                slice_ = [slice(None)]
                slice_.extend(tuple(
                    slice(None) if i == spatial_dimension_i else None for i in range(len(data_shape))
                ))

                rep = [1]
                rep.extend(tuple(
                    1 if i == spatial_dimension_i else data_shape[i] for i in range(len(data_shape))
                ))

                sin_xx = sin_x[slice_].repeat(*rep)
                cos_xx = cos_x[slice_].repeat(*rep)

                additive = torch.cat((sin_xx, cos_xx), dim=0)

                encoding_slice = [slice(None),
                                  slice(spatial_dimension_i * self.encoding_dimension * 2,
                                        spatial_dimension_i * self.encoding_dimension * 2 +
                                        2 * self.encoding_dimension),
                                  ]
                encoding_slice.extend([slice(None) for _ in range(len(data_shape))])

                encoding[encoding_slice] = additive

                self.cache = encoding
        return self.cache

    def get_encoding_shape(self, batch: torch.Tensor):
        return torch.Size((batch.shape[0],
                           2 * (len(batch.shape) - 2) * self.encoding_dimension,
                           *batch.shape[2:]))

    def forward(self, input_tensor: torch.Tensor):
        encoding = self.encode(input_tensor)
        return torch.cat((input_tensor, encoding), dim=1)
