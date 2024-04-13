# %%
import torch as th
import torch.nn as nn


# %%
# embedding layer for the teams

class TeamEmbedding(nn.Module):
    def __init__(self, num_teams, embedding_dim):
        super(TeamEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_teams, embedding_dim)
    
    def forward(self, x):
        # x shape = (batch_size, num_features, time_steps)

        home_idx = x[:, 0, :].int() # shape = (batch_size, time_steps)
        away_idx = x[:, 1, :].int() # shape = (batch_size, time_steps)

        home_embedding = self.embedding(home_idx) # shape = (batch_size, time_steps, embedding_dim)
        away_embedding = self.embedding(away_idx) # shape = (batch_size, time_steps, embedding_dim)

        # concatenate the home and away embeddings along the last dimension
        home_and_away_embedding = th.cat((home_embedding, away_embedding), dim=-1).permute(0, 2, 1) # shape = (batch_size, 2*embedding_dim, time_steps)
        x = th.concatenate((home_and_away_embedding, x[:, 2:, :]), dim=1)
        
        return x
    
# %%
class ConvBlock2D(nn.Module):
    """
    Reusable Convolutional Block

    Args: 
        in_channels: int: Number of input channels
        out_channels: int: Number of output channels
        kernel_size: int: Size of the kernel
        stride: int: Stride of the kernel
        padding: int: Padding of the kernel
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock2D, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # batch normalization for the output of the convolutional layer
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# %%

# code from: https://medium.com/@karuneshu21/implement-inception-v1-in-pytorch-66bdbb3d0005

class InceptionBlock2D(nn.Module):
    """
    Reusable Inception Block
    Has 4 parallel convolutional blocks:
        1x1 Convolution
        1x1 Convolution followed by 3x3 Convolution
        1x1 Convolution followed by 5x5 Convolution
        3x3 Max2d Pooling followed by 1x1 Convolution

         Note:
            1. output and input feature map height and width should remain the same. Only the channel output should change. eg. 28x28x192 -> 28x28x256
            2. To generate same height and width of output feature map as the input feature map, following should be padding for
                - 1x1 conv : p=0
                - 3x3 conv : p=1
                - 5x5 conv : p=2

    Args:
       in_channels (int) : # of input channels
       out_1x1 (int) : number of output channels for branch 1
       red_3x3 (int) : reduced 3x3 referring to output channels of 1x1 conv just before 3x3 in branch2
       out_3x3 (int) : number of output channels for branch 2
       red_5x5 (int) : reduced 5x5 referring to output channels of 1x1 conv just before 5x5 in branch3
       out_5x5 (int) : number of output channels for branch 3
       out_1x1_pooling (int) : number of output channels for branch 4

    Attributes:
        concatenated feature maps from all 4 branches constituiting output of Inception module.
    """
    def __init__(self , in_channels , out_1x1 , red_3x3 , out_3x3 , red_5x5 , out_5x5 , out_1x1_pooling):
        super(InceptionBlock2D , self).__init__()
        # Branch 1    kernel 1x1    stride 1    padding 0
        self.branch1 = ConvBlock2D(in_channels , out_1x1 , kernel_size=1 , stride=1 , padding=0)

        # Branch 2    kernel 1x1    stride 1    padding 0
        #             kernel 3x3    stride 1    padding 1
        self.branch2 = nn.Sequential(
            ConvBlock2D(in_channels , red_3x3 , kernel_size=1 , stride=1 , padding=0),
            ConvBlock2D(red_3x3 , out_3x3 , kernel_size=3 , stride=1 , padding=1)
        )

        # Branch 3    kernel 1x1    stride 1    padding 0
        #             kernel 5x5    stride 1    padding 2
        self.branch3 = nn.Sequential(
            ConvBlock2D(in_channels , red_5x5 , kernel_size=1 , stride=1 , padding=0),
            ConvBlock2D(red_5x5 , out_5x5 , kernel_size=5 , stride=1 , padding=2)
        )

        # Branch 4    kernel 3x3    stride 1    padding 1
        #             kernel 1x1    stride 1    padding 0
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3 , stride=1 , padding=1),
            ConvBlock2D(in_channels , out_1x1_pooling , kernel_size=1 , stride=1 , padding=0)
        )

    def forward(self , x):
        # x should have shape [batch_size , in_channels , height , width]
        return th.cat([self.branch1(x) , self.branch2(x) , self.branch3(x) , self.branch4(x)] , 1)




# %%
class Inceptionv1(nn.Module):
    """
    Args:
        in_channels (int) : # of input channels (1 for time series data)
        out_channels (int) : # of output channels

        There is no number of classes since the output is input to another layer
    """
    def __init__(self , in_channels = 1):
        super(Inceptionv1,self).__init__()

        # in_channels , out_channels , kernel_size , stride , padding
        self.conv1 = ConvBlock2D(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Sequential(
            ConvBlock2D(64, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock2D(64, 192, kernel_size=3, stride=1, padding=1)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3 , stride=1 , padding=1)

        self.inception3a = InceptionBlock2D(in_channels=192 , out_1x1=64 , red_3x3=96 , out_3x3=128 , red_5x5=16 , out_5x5=32 , out_1x1_pooling=32)
        self.inception3b = InceptionBlock2D(in_channels=256 , out_1x1=128 , red_3x3=128 , out_3x3=192 , red_5x5=32 , out_5x5=96 , out_1x1_pooling=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)

        self.inception4a = InceptionBlock2D(in_channels=480 , out_1x1=192 , red_3x3=96 , out_3x3=208 , red_5x5=16 , out_5x5=48 , out_1x1_pooling=64)
        self.inception4b = InceptionBlock2D(in_channels=512 , out_1x1=160 , red_3x3=112 , out_3x3=224 , red_5x5=24 , out_5x5=64 , out_1x1_pooling=64)
        self.inception4c = InceptionBlock2D(in_channels=512 , out_1x1=128 , red_3x3=128 , out_3x3=256 , red_5x5=24 , out_5x5=64 , out_1x1_pooling=64)
        self.inception4d = InceptionBlock2D(in_channels=512 , out_1x1=112 , red_3x3=144 , out_3x3=288 , red_5x5=32 , out_5x5=64 , out_1x1_pooling=64)
        self.inception4e = InceptionBlock2D(in_channels=528 , out_1x1=256 , red_3x3=160 , out_3x3=320 , red_5x5=32 , out_5x5=128 , out_1x1_pooling=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)
        
        self.inception5a = InceptionBlock2D(in_channels=832 , out_1x1=256 , red_3x3=160 , out_3x3=320 , red_5x5=32 , out_5x5=128 , out_1x1_pooling=128)
        self.inception5b = InceptionBlock2D(in_channels=832 , out_1x1=384 , red_3x3=192 , out_3x3=384 , red_5x5=48 , out_5x5=128 , out_1x1_pooling=128)

        self.avgpool = nn.AvgPool2d(kernel_size=7 , stride=1)

    def forward(self , x):
        # x should have shape [batch_size , in_channels , height , width]
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        # Output shape [batch_size , 1024 , 1 , 1]
        # This mode is very slow but much more complex
        return x



# %%
class Inceptionv2(nn.Module):
    """
    Args:
        in_channels (int) : Number of input channels (1 for time series data transformed into a compatible format)
    """
    def __init__(self, in_channels=1):
        super(Inceptionv2, self).__init__()

        # Initial convolution and maxpool layers
        self.conv1 = ConvBlock2D(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Reduced convolution sequence
        self.conv2 = nn.Sequential(
            ConvBlock2D(64, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock2D(64, 192, kernel_size=3, stride=1, padding=1)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Selective Inception blocks
        self.inception3a = InceptionBlock2D(in_channels=192, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1_pooling=32)
        self.inception3b = InceptionBlock2D(in_channels=256, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1_pooling=64)
        self.inception4a = InceptionBlock2D(in_channels=480, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1_pooling=64)

        # Final pooling layer
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception4a(x)
        x = self.avgpool(x)
        # out put shape [batch_size , 512 , 1 , 1]
        return x
    
# %%

class Inceptionv3(nn.Module):
    """
    Very simplified Inception model with only one Inception layer.
    Designed to process input with a single channel, suitable for modified time series data.
    
    Args:
        in_channels (int) : Number of input channels (1 for time series data transformed into a compatible format)
    """
    def __init__(self, in_channels=1):
        super(Inceptionv3, self).__init__()

        # Initial convolution and maxpool layers
        self.conv1 = ConvBlock2D(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Reduced convolution sequence
        self.conv2 = nn.Sequential(
            ConvBlock2D(64, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock2D(64, 192, kernel_size=3, stride=1, padding=1)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Only one Inception block
        self.inception3a = InceptionBlock2D(in_channels=192, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1_pooling=32)

        # Final pooling layer to summarize the features
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

    def forward(self, x):
        # x should have shape [batch_size , in_channels , height , width]
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.avgpool(x)
        # out put shape [batch_size , 256 , 1 , 1]
        return x
    
# %%
# inception block
# class InceptionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, num_kernel=4, init_weights=True):
#         super(InceptionBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_kernel = num_kernel
#         kernels = []
#         for i in range(1, num_kernel + 1):
#             # kernels.append(nn.Conv2d(in_channels, out_channels // num_kernel, kernel_size=i * 2 + 1, padding=i))
#             kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=i * 2 + 1, padding=i))
#         self.kernels = nn.ModuleList(kernels)
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         outputs = []
#         for i in range(self.num_kernel):
#             outputs.append(self.kernels[i](x))
#         # outputs = th.stack(outputs, dim=1)
#         outputs = th.cat(outputs, dim=1).mean(dim=1)

#         return outputs

# %%