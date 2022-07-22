import torch
import torchvision.models as models
import torch.nn as nn

from torchvision.models.resnet import Bottleneck
from . mish import Mish


class ResnetRegressor(torch.nn.Module):
    def __init__(self, widths: tuple = (512, 128), first_conv_in: int = 3, first_conv_out: int = 128):
        super(ResnetRegressor, self).__init__()
        self.resnet = models.resnet50(pretrained=False, progress=False)

        self.resnet.conv1 = torch.nn.Conv2d(first_conv_in, first_conv_out,
                                            kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.bn1 = torch.nn.BatchNorm2d(first_conv_out,
                                               eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # kind of dirty hack, needs as there is no way to pass to ResNet constructor it's self.inplanes variable
        # there is magic numbers 64 and 256 which took from ResNet50 definition
        self.resnet.layer1[0].conv1 = nn.Conv2d(
            first_conv_out, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.resnet.layer1[0].downsample[0] = nn.Conv2d(
            first_conv_out, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.resnet.fc = torch.nn.Identity()
        self.set_train_convolutional_part(False)

        # todo in_features=2048 replace by something like
        # self.resnet.fc.output_shape
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
