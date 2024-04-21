import torch
import torch.nn as nn

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, init_weight=True, use_bias=False):
        super(ResidualLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=use_bias)

        if init_weight:
            nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
            if self.conv1x1.bias is not None:
                nn.init.constant_(self.conv1x1.bias, 0)

    def forward(self, x, output):
        x = self.conv1x1(x)
        return torch.add(x, output)