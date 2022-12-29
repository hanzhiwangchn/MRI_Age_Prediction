import math
import torch
import torch.nn as nn
from torchsummary.torchsummary import summary


# ------------------- ResNet ---------------------

class ResNetBlock(nn.Module):
    def __init__(self, block_number, input_size):
        super(ResNetBlock, self).__init__()

        layer_in = input_size if input_size is not None else 2 ** (block_number + 1)
        layer_in = layer_in if block_number < 4 else 32
        layer_out = 2 ** (block_number + 2) if block_number < 4 else layer_in

        self.conv1 = nn.Conv2d(layer_in, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_out)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(layer_out, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(layer_out)
        self.shortcut = nn.Conv2d(layer_in, layer_out, kernel_size=1, stride=1, bias=False)
        self.act2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_size):
        super(ResNet, self).__init__()

        self.layer1 = self._make_block(1, input_size[0])
        self.layer2 = self._make_block(2)
        self.layer3 = self._make_block(3)
        self.layer4 = self._make_block(4)
        self.layer5 = self._make_block(5)

        h, w = ResNet._maxpool_output_size(input_size[1::], nb_layers=5)

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(32 * h * w, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    @staticmethod
    def _make_block(block_number, input_size=None):
        return nn.Sequential(
            ResNetBlock(block_number, input_size),
            nn.MaxPool2d(2, stride=2))

    @staticmethod
    def _maxpool_output_size(input_size, kernel_size=(2, 2), stride=(2, 2), nb_layers=1):
        h = math.floor((input_size[0] - kernel_size[0]) / stride[0] + 1)
        w = math.floor((input_size[1] - kernel_size[1]) / stride[1] + 1)

        if nb_layers == 1:
            return h, w
        return ResNet._maxpool_output_size(input_size=(h, w), kernel_size=kernel_size,
                                           stride=stride, nb_layers=nb_layers - 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc(out)
        return out


# ------------------- ResNet for ABIDE----------------

class ResNetBlock_stride(nn.Module):
    def __init__(self, block_number, input_size):
        super(ResNetBlock_stride, self).__init__()

        layer_in = input_size if input_size is not None else 2 ** (block_number + 1)
        layer_out = 2 ** (block_number + 2) if block_number < 5 else 64

        self.conv1 = nn.Conv3d(layer_in, layer_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(layer_out)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv3d(layer_out, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(layer_out)
        self.shortcut = nn.Conv3d(layer_in, layer_out, kernel_size=1, stride=2, bias=False)
        self.act2 = nn.ELU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.act2(out)
        return out


class ResNet_stride(nn.Module):
    def __init__(self, input_size):
        super(ResNet_stride, self).__init__()

        self.layer1 = self._make_block(1, input_size[0])
        self.layer2 = self._make_block(2)
        self.layer3 = self._make_block(3)
        self.layer4 = self._make_block(4)
        self.layer5 = self._make_block(5)

        self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(64 * 2 * 2 * 2, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    @staticmethod
    def _make_block(block_number, input_size=None):
        return nn.Sequential(
            ResNetBlock_stride(block_number, input_size))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = self.fc(out)
        return out


# ------------------- VGG ---------------------

class VGG(nn.Module):
    def __init__(self, input_size):
        super(VGG, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(input_size[0], 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(3, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(3, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(3, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(3, 2),

            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(3, 2),
        )

        d, h, w = VGG._maxpool_output_size(input_size[1::], nb_layers=5)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(64 * d * h * w, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def _maxpool_output_size(input_size, kernel_size=(3, 3, 3), stride=(2, 2, 2), nb_layers=1):
        d = math.floor((input_size[0] - kernel_size[0]) / stride[0] + 1)
        h = math.floor((input_size[1] - kernel_size[1]) / stride[1] + 1)
        w = math.floor((input_size[2] - kernel_size[2]) / stride[2] + 1)

        if nb_layers == 1:
            return d, h, w
        return VGG._maxpool_output_size(input_size=(d, h, w), kernel_size=kernel_size,
                                        stride=stride, nb_layers=nb_layers - 1)


# ------------------- VGG for ABIDE---------------------

class VGG_stride(nn.Module):
    def __init__(self, input_size):
        super(VGG_stride, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(input_size[0], 8, 3, padding=1, stride=2),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1, stride=2),
            nn.BatchNorm3d(8),
            nn.ReLU(),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(64 * 3 * 2 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------- Inception ---------------------

class Inception_Block(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception_Block, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm3d(n1x1),
            nn.ReLU(True),
        )
        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm3d(n3x3red),
            nn.ReLU(True),
            nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3x3),
            nn.ReLU(True),
        )
        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv3d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm3d(n5x5red),
            nn.ReLU(True),
            nn.Conv3d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(True),
            nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(True),
        )
        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm3d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class Inception(nn.Module):
    def __init__(self, input_size):
        super(Inception, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.MaxPool3d(3, stride=2),

            nn.Conv3d(8, 8, kernel_size=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.MaxPool3d(3, stride=2),
        )

        self.a3 = Inception_Block(16, 12, 8, 12, 4, 4, 4)
        self.b3 = Inception_Block(32, 24, 16, 24, 6, 8, 8)
        self.a4 = Inception_Block(64, 48, 32, 48, 12, 16, 16)
        self.b4 = Inception_Block(128, 72, 48, 72, 18, 24, 24)
        self.a5 = Inception_Block(192, 96, 72, 96, 24, 32, 32)

        self.maxpool = nn.MaxPool3d(3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(256 * 2 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.avgpool(out)
        out = self.classifier(out)
        return out


# ------------------- Inception for ABIDE ---------------------

class Inception_Block_stride(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception_Block_stride, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(in_planes, n1x1, kernel_size=1, stride=2),
            nn.BatchNorm3d(n1x1),
            nn.ReLU(True),
        )
        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm3d(n3x3red),
            nn.ReLU(True),
            nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(n3x3),
            nn.ReLU(True),
        )
        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv3d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm3d(n5x5red),
            nn.ReLU(True),
            nn.Conv3d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(True),
            nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(True),
        )
        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool3d(3, stride=2, padding=1),
            nn.Conv3d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm3d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class Inception_stride(nn.Module):
    def __init__(self, input_size):
        super(Inception_stride, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(8),
            nn.ReLU(True),

            nn.Conv3d(8, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
        )

        self.a3 = Inception_Block_stride(16, 12, 8, 12, 4, 4, 4)
        self.b3 = Inception_Block_stride(32, 24, 16, 24, 6, 8, 8)
        self.a4 = Inception_Block_stride(64, 48, 32, 48, 12, 16, 16)

        self.maxpool = nn.MaxPool3d(3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 2 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.a4(out)
        out = self.avgpool(out)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    input_size = (3, 100, 100)
    model = ResNet(input_size=input_size)
    summary(model, input_size)
