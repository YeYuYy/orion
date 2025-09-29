'''UNet in PyTorch.

Reference:
[1] Olaf Ronneberger, Philipp Fischer, Thomas Brox
    U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv:1505.04597
'''
import torch
import torch.nn as nn
import orion.nn as on


class DoubleConv(on.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            on.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            on.BatchNorm2d(out_channels),
            on.ReLU(),
            on.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            on.BatchNorm2d(out_channels),
            on.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(on.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = on.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        out = self.pool(skip)
        return out, skip


class UpBlock(on.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = on.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.cat = on.Cat()

    def forward(self, x, skip):
        x = self.up(x)
        x = self.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(on.Module):
    def __init__(self, in_channels=3, num_classes=10, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder (contracting path)
        for feature in features:
            self.downs.append(DownBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (expansive path)
        for feature in reversed(features):
            self.ups.append(UpBlock(feature * 2, feature))

        # Final classifier
        self.final_conv = on.Conv2d(features[0], num_classes, kernel_size=1)
        self.avgpool = on.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.ups)):
            x = self.ups[idx](x, skip_connections[idx])

        # Final classification
        x = self.final_conv(x)
        x = self.avgpool(x)
        return x


def UNet_small():
    return UNet(features=[32, 64, 128, 256])


def UNet_base():
    return UNet(features=[64, 128, 256, 512])


def UNet_large():
    return UNet(features=[96, 192, 384, 768])


def test():
    net = UNet_base()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
