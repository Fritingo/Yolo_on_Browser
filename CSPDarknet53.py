
import torch
import torch.nn as nn
import torch.nn.functional as F

# model structure yolov4 https://dl.acm.org/doi/pdf/10.1145/3478905.3478951

# Mish activation func
class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x): # x * tanh(ln(1 + e ^ x))
        return x * torch.tanh(F.softplus(x))

# Basic Conv Block
# Conv -> BN -> Mish
# args (in_channels, out_channels, kernel_size, stride)
class BasicConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(BasicConv,self).__init__()
        # Conv in , out , kernel size , stride , padding , bias
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,kernel_size//2,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Mish()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

# Residual Block
# ---> Basic Conv (kernel size 1) -> Basic Conv (kernel size 3) - + ->
#  |                                                              ^
#  ---------------------------------------------------------------|
class ResBlock(nn.Module):
    def __init__(self,in_channels,hidden_channels = None):
        super(ResBlock,self).__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        self.block = nn.Sequential(
            BasicConv(in_channels,hidden_channels,1,1), # size same
            BasicConv(hidden_channels,in_channels,3,1) # size same
        )

    def forward(self,x):
        return x + self.block(x)


# CSP Residual Block         < conv block >       < conv concat >  
# downsample ---> Conv0 ---> Conv ---> Conv ---> + ---> Conv ---> concat --->
#             |         |                       ^                  ^
#             |         ------------------------|                  |
#             --> Conv1 --------------------------------------------| 

class FirstCSPBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(FirstCSPBlock,self).__init__()
        # downsampleb  reduce size
        self.downsample = BasicConv(in_channels,out_channels,3,2)
        
        self.split_conv0 = BasicConv(out_channels,out_channels,1,1)
        self.split_conv1 = BasicConv(out_channels,out_channels,1,1)

        self.conv_block = nn.Sequential(
            ResBlock(out_channels,out_channels//2),
            BasicConv(out_channels,out_channels,1,1)
        )

        self.conv_concat = BasicConv(out_channels*2,out_channels,1,1)

    def forward(self,x):
        x = self.downsample(x)
        # print('downsampe ',x.shape)
        x0 = self.split_conv0(x)
        
        x1 = self.split_conv1(x)
        # print('x0:',x0.shape,'x1:',x1.shape)

        x0 = self.conv_block(x0)
        # print('x0 2',x0.shape)

        x = torch.cat([x0,x1],dim=1)
        # print('combine ',x.shape)
        x = self.conv_concat(x)

        return x


class CSPBlock(nn.Module):
    def __init__(self, in_channels,out_channels,num_blocks):
        super(CSPBlock,self).__init__()
        # downsampleb  reduce size
        self.downsample = BasicConv(in_channels,out_channels,3,2)
        
        self.split_conv0 = BasicConv(out_channels,out_channels//2,1,1)
        self.split_conv1 = BasicConv(out_channels,out_channels//2,1,1)

        self.conv_block = nn.Sequential(
            *[ResBlock(out_channels//2,out_channels//2) for _ in range(num_blocks)],
            BasicConv(out_channels//2,out_channels//2,1,1)
        )

        self.conv_concat = BasicConv(out_channels,out_channels,1,1)

    def forward(self,x):
        x = self.downsample(x)
        # print('2 downsample',x.shape)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        # print('2 x0:',x0.shape,'2 x1:',x1.shape)

        x0 = self.conv_block(x0)
        # print('2 x0:',x0.shape)

        x = torch.cat([x0,x1],dim=1)
        x = self.conv_concat(x)

        return x

# CSPDarknet53
# input 416x416x3
# output 
class CSPDarknet(nn.Module):
    def __init__(self,num_feature):
        super(CSPDarknet,self).__init__()

        self.num_feature = num_feature

        self.conv1 = BasicConv(3,32,3,1)
        self.test = FirstCSPBlock(32, 64)
        self.test2 = CSPBlock(64,128,2)
        self.stages = nn.ModuleList([
            # 416,416,32 -> 208,208,64
            FirstCSPBlock(32, 64),
            # 208,208,64 -> 104,104,128
            CSPBlock(64, 128, 2),
            # 104,104,128 -> 52,52,256
            CSPBlock(128, 256, 8),
            # 52,52,256 -> 26,26,512
            CSPBlock(256, 512, 8),
            # 26,26,512 -> 13,13,1024
            CSPBlock(512, 1024, 4)
        ])

    def forward(self,x):
        x = self.conv1(x)
        # print('one---> ',x.shape)
        feature = []
        for stage in self.stages:
            x = stage(x)
            # print('stage---> ',x.shape)
            feature.append(x)

        return feature[-self.num_feature:]
     


if __name__ == '__main__':
    model = CSPDarknet(2)
    x = torch.randn(1,3,416,416)
    y = model(x)
    print(len(y),y)