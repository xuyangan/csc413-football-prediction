import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionTimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, init_weight=True, bottleneck_dim=None):
        super(InceptionTimeBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck = bottleneck_dim
        
        self.bottleneck = None
        # If bottleneck is specified, reduce the dimsnsionality of data
        if bottleneck_dim is not None:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_dim, kernel_size=1)
            in_channels = bottleneck_dim

        # 1x1 Convolution
        self.branch1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # 3x3 Convolution, padding = 1 to keep the size the same
        self.branch3x3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        # 5x5 Convolution, padding = 2 to keep the size the same
        self.branch5x5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)

        # Max pooling
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # 1x1 Convolution after max pooling
        self.max_pool_branch1x1 = nn.Conv1d(self.in_channels, out_channels, kernel_size=1)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        branch_max_pool_1x1 = self.max_pool(x)
        branch_max_pool_1x1 = self.max_pool_branch1x1(x)

        _x = x
        if self.bottleneck is not None:
            _x = self.bottleneck(x)
        
        branch_1x1 = self.branch1x1(_x)
        branch_3x3 = self.branch3x3(_x)
        branch_5x5 = self.branch5x5(_x)

        outputs = [branch_1x1, branch_3x3, branch_5x5, branch_max_pool_1x1]
        outputs = torch.cat(outputs, dim=1)

        return F.relu(outputs).permute(0, 2, 1)