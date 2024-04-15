# %%
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# %%



class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        # Retrieve features and label at the specified index
        return self.features[index], self.labels[index]
    
# %%

class ConvBlock1D(nn.Module):
    """
    Reusable 1D Convolutional Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock1D, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# %%

class InceptionBlock1D(nn.Module):
    """
    Reusable Inception Block using 1D convolutions
    """
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pooling):
        super(InceptionBlock1D, self).__init__()
        
        # Branch 1: 1x1 Convolution
        self.branch1 = ConvBlock1D(in_channels, out_1x1, kernel_size=1, stride=1, padding=0)

        # Branch 2: 1x1 followed by 3x3 Convolution
        self.branch2 = nn.Sequential(
            ConvBlock1D(in_channels, red_3x3, kernel_size=1, stride=1, padding=0),
            ConvBlock1D(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1)
        )

        # Branch 3: 1x1 followed by 5x5 Convolution
        self.branch3 = nn.Sequential(
            ConvBlock1D(in_channels, red_5x5, kernel_size=1, stride=1, padding=0),
            ConvBlock1D(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2)
        )

        # Branch 4: 3x3 Max Pooling followed by 1x1 Convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            ConvBlock1D(in_channels, out_1x1_pooling, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # Concatenate the outputs of each branch along the channel dimension
        return th.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

# %%

class Inceptionv3(nn.Module):
    """
    Very simplified Inception model with only one Inception layer using 1D convolutions.
    Designed to process input with a single channel, suitable for modified time series data.
    
    Args:
        in_channels (int): Number of input channels (1 for time series data transformed into a compatible format)
    """
    def __init__(self, in_channels=1):
        super(Inceptionv3, self).__init__()

        # Initial convolution and maxpool layers using 1D operations
        self.conv1 = ConvBlock1D(in_channels, 64, kernel_size=3, stride=1, padding=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Reduced convolution sequence
        self.conv2 = nn.Sequential(
            ConvBlock1D(64, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock1D(64, 192, kernel_size=3, stride=1, padding=1)
        )
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # Only one Inception block
        self.inception3a = InceptionBlock1D(in_channels=192, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1_pooling=32)

        # Final pooling layer to summarize the features using 1D pooling
        self.avgpool = nn.AvgPool1d(kernel_size=7, stride=1)

    def forward(self, x):
        # x should have shape [batch_size , in_channels , length]
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.avgpool(x)
        # Output shape [batch_size, 256, 1], depending on input length and convolutions
        return x
