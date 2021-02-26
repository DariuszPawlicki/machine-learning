import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, k=2, n=5, alpha=1e-4, beta=0.75):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 11, 4) # Input Channels, Output Channels, Kernel Size, Stride, Padding

        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv2.bias = torch.nn.Parameter(torch.ones_like(self.conv2.bias))

        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)

        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv4.bias = torch.nn.Parameter(torch.ones_like(self.conv4.bias))

        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.conv5.bias = torch.nn.Parameter(torch.ones_like(self.conv5.bias))

        self.fc1 = nn.Linear(9216, 4096)
        self.fc1.bias = torch.nn.Parameter(torch.ones_like(self.fc1.bias))

        self.fc2 = nn.Linear(4096, 4096)
        self.fc2.bias = torch.nn.Parameter(torch.ones_like(self.fc2.bias))

        self.fc3 = nn.Linear(4096, 1000)
        self.fc3.bias = torch.nn.Parameter(torch.ones_like(self.fc3.bias))

        self.pooling = nn.MaxPool2d(3, 2)
        self.local_norm = nn.LocalResponseNorm(n, alpha, beta, k)
    

    def forward(self, x):
        x = self.local_norm(self.conv1(x))
        x = self.pooling(F.relu(x))

        x = self.local_norm(self.conv2(x))
        x = self.pooling(F.relu(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.conv5(x)
        x = self.pooling(F.relu(x))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)

        x = F.relu(self.fc3(x))

        return x