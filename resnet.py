import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shortcuts, downsampling=False):
        super(ResidualBlock, self).__init__()

        self.downsampling = downsampling
        self.shortcuts = shortcuts

        if downsampling == True:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    

    def forward(self, x):
        if self.shortcuts == True:
            residual = torch.empty_like(x).copy_(x)

        x = self.bn1(self.conv1(x))
        x = F.relu(x)

        x = self.bn2(self.conv2(x))
        x = F.relu(x)

        if self.shortcuts == True:
            if self.downsampling == True:
                residual = nn.MaxPool2d(1, 2)(residual)
                padding = torch.zeros_like(residual)

                residual = torch.cat((residual, padding), dim=1)

            x += residual

        return x


class ResNet(nn.Module):
    def __init__(self, size, shortcuts=True):
        """
        Resnet implementation for CIFAR-10 dataset.\n
        Size parameter in constructor is 'n' from network size formula: 6n + 2.
        Shortuct parameter tells, that this network will use a residual connections,
        if False then model will become plain Convolutional Neural Network.
        """
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)

        self.residuals = []

        filters_count = [16, 32, 64]

        for i, filters in enumerate(filters_count): # Network contains layers with three different
                                                    # feature map sizes: 32x32, 16x16, 8x8 and filters count

            for j in range(size):      # Each part of network with certain feature maps size
                                       # contains 'n' residual blocks what gives '2n' layers for each
                                       # feature map size.
                if i != 0 and j == 0:
                    """
                    If first layer of downsampled feature maps,
                    than downsampling map size by 2 and multiply
                    filters count by 2.
                    """
                    self.residuals.append(ResidualBlock(filters_count[i-1], filters, 
                                                        shortcuts=shortcuts, downsampling=True))
                else:
                    self.residuals.append(ResidualBlock(filters, filters, shortcuts=shortcuts))
        
        self.residuals = nn.Sequential(*self.residuals)

        self.global_pooling = nn.AvgPool2d(8)
        self.out = nn.Linear(64, 1000) 


    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)

        x = self.residuals(x)

        x = self.global_pooling(x)
        x = torch.reshape(x, (x.shape[0], 64))

        x = self.out(x)

        return x