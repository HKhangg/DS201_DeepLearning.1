import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(
            self, 
            channels: int, 
            c1x1: int,
            c3x3_reduced: int,
            c3x3: int,
            c5x5_reduced: int,
            c5x5: int,
            pool_proj: int
    ) -> None:
        super().__init__()

        self.branch1 = conv_block(channels, c1x1, 1, padding=0)
        
        self.branch2 = nn.Sequential(
            conv_block(channels, c3x3_reduced, 1, padding=0),
            conv_block(c3x3_reduced, c3x3, 3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            conv_block(channels, c5x5_reduced, 1, padding=0),
            conv_block(c5x5_reduced, c5x5, 5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            conv_block(channels, pool_proj, 1, padding=0)
        )

    def forward(self, x: torch.Tensor):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # Initial convolution layers
        self.pre_inception = nn.Sequential(
            conv_block(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            conv_block(64, 64, 1),
            conv_block(64, 192, 3, padding=1),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )

        # Inception 3
        self.inception3 = nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )

        # Inception 4
        self.inception4 = nn.Sequential(
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )

        # Inception 5
        self.inception5 = nn.Sequential(
            InceptionBlock(832, 256, 160, 320, 32, 128, 128),
            InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.pre_inception(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        return self.classifier(x)