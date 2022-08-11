import torch
import torchvision.models as models
import torch.nn as nn

from .mish import Mish
from .positional_encoding import SinusoidalPositionalEmbedding


class ResnetRegressor(torch.nn.Module):
    def __init__(self, widths: tuple = (512, 128), encoder_dimension: int = 2, first_conv_out: int = 128):
        super(ResnetRegressor, self).__init__()
        self.encoder_dimension = encoder_dimension
        self.resnet = models.resnet50(pretrained=False, progress=False)

        self.resnet.conv1 = CoordConv2d(
            self.encoder_dimension,
            1, first_conv_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.bn1 = torch.nn.BatchNorm2d(
            first_conv_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # kind of dirty hack, needs as there is no way to pass to ResNet constructor it's self.inplanes variable
        # there is magic numbers 64 and 256 which took from ResNet50 definition
        self.resnet.layer1[0].conv1 = torch.nn.Conv2d(
            first_conv_out, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.resnet.layer1[0].downsample[0] = torch.nn.Conv2d(
            first_conv_out, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # # change all Conv2d to CoordConv2d
        # for layer in (
        #         self.resnet.layer1,
        #         self.resnet.layer2,
        #         self.resnet.layer3,
        #         self.resnet.layer4,):
        #     for bottleneck in layer:
        #         for conv_name in ('conv1', 'conv2', 'conv3'):
        #             conv = bottleneck.__getattr__(conv_name)
        #
        #             if isinstance(conv, torch.nn.Conv2d):
        #                 # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None
        #                 bottleneck.__setattr__(conv_name, CoordConv2d(
        #                     self.encoder_dimension,
        #                     conv.in_channels, conv.out_channels,
        #                     kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding,
        #                     dilation=conv.dilation, groups=conv.groups, bias=conv.bias, padding_mode=conv.padding_mode))

        # insert CoordConv2d between ResNet Bottleneck blocks
        for layer_str in (
                    'layer1',
                    'layer2',
                    'layer3',
                    'layer4',):
            original_layer = self.resnet.__getattr__(layer_str)
            new_layer = torch.nn.Sequential()
            for bottleneck in original_layer:
                new_layer.append(bottleneck)

                new_layer.append(CoordConv2d(
                    self.encoder_dimension,
                    bottleneck.conv3.out_channels, bottleneck.conv3.out_channels, kernel_size=3, stride=1, padding=1,
                ))

            self.resnet.__setattr__(layer_str, new_layer)

        self.resnet.fc = torch.nn.Identity()
        self.set_train_convolutional_part(False)

        # todo in_features=2048 replace by something like
        # self.resnet.layer4[2].conv3.out_channels
        self.widths = [2048]
        self.widths.extend(widths)
        self.widths.append(1)

        layers = []
        for i in range(len(self.widths) - 1):
            layers.append(nn.Linear(in_features=self.widths[i], out_features=self.widths[i + 1]))
            layers.append(Mish())

        self.fully_connected = torch.nn.Sequential(*layers)

    def forward(self, image_batch):
        features = self.resnet(image_batch)
        result = self.fully_connected(features)
        return result

    def set_train_convolutional_part(self, value: bool):
        for param in self.resnet.parameters():
            param.requires_grad = value


class CoordConv2d(torch.nn.Module):
    def __init__(self, encoder_dimension: int, *args, **kwargs):
        super().__init__()
        self.encoder_dimension = encoder_dimension
        # 4 because sin_cos x_y
        args = list(args)
        args[0] += encoder_dimension * 4
        self.conv = torch.nn.Conv2d(*args, **kwargs)
        self.encoder = SinusoidalPositionalEmbedding(encoder_dimension)

    def forward(self, input_tensor: torch.Tensor):
        input_tensor = self.encoder(input_tensor)
        return self.conv(input_tensor)
