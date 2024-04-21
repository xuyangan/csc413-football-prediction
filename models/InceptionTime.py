import torch.nn as nn
import torch.nn.functional as F

from layers.InceptionTimeBlock import InceptionTimeBlock
from layers.ResidualLayer import ResidualLayer

class InceptionTime(nn.Module):
    def __init__(self, in_channels, out_channels, depth=6, init_weight=True, bottleneck_dim=None, use_bias=False, use_residual=True):
        super(InceptionTime, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck = bottleneck_dim

        blocks = []
        residual_in_channels = in_channels
        for i in range(depth):
            bottleneck_dim = self.bottleneck

            if self.bottleneck is not None and self.bottleneck >= in_channels:
                bottleneck_dim = None
            
            blocks.append(InceptionTimeBlock(in_channels, 
                                             out_channels, 
                                             init_weight=init_weight,
                                             bottleneck_dim=bottleneck_dim,
                                             use_bias=False))
            
            if use_residual and i % 2 == 1:
                blocks.append(ResidualLayer(residual_in_channels, out_channels * 4))
                residual_in_channels = out_channels * 4
            
            in_channels = out_channels * 4
            out_channels = in_channels // 2

        self.blocks = nn.ModuleList(blocks)


    def forward(self, x):
        # Change dimension of x: [b, t, d/c] -> [b, d/c, t]
        x = x.permute(0, 2, 1)
        output = x
        for block in self.blocks:
            if isinstance(block, InceptionTimeBlock):
                output = F.relu(block(output))
            else:
                output = F.relu(block(x, output))
                x = output

        return output.permute(0, 2, 1)