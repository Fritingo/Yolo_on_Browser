
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from CSPDarknet53 import CSPDarknet

# SPP

class SSP(nn.Module):
    def __init__(self,pool_sizes=[5,9,13]):
        super(SSP,self).__init__()

        self.maxpool0 = nn.MaxPool2d(pool_sizes[0],1,pool_sizes[0]//2)
        self.maxpool1 = nn.MaxPool2d(pool_sizes[1],1,pool_sizes[1]//2)
        self.maxpool2 = nn.MaxPool2d(pool_sizes[2],1,pool_sizes[2]//2)

    def forward(self,x):
        feature0 = self.maxpool0(x)
        # print('0',feature0.shape)
        feature1 = self.maxpool1(x)
        # print('1',feature1.shape)
        feature2 = self.maxpool2(x)
        # print('2',feature2.shape)
        # print('x',x.shape)

        features = torch.cat((feature0,feature1,feature2,x),dim=1)

        return features

# Conv operation
class conv_op(nn.Module):
    def __init__(self,in_filters,out_filters,kernel_size,stride=1):
        super(conv_op,self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_filters,out_filters,kernel_size,stride,padding=0,bias=False),
            nn.BatchNorm2d(out_filters),
            nn.LeakyReLU(0.1),
        )
    
    def forward(self,x):
        return self.conv(x)


# 3 Conv
class conv_3(nn.Module):
    def __init__(self,in_filters,mid_filters,out_filters):
        super(conv_3,self).__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_filters,mid_filters,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(mid_filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_filters,out_filters,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_filters,mid_filters,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(mid_filters),
            nn.LeakyReLU(0.1),
        )

    def forward(self,x):
        return self.conv3(x)

# 5 Conv
class conv_5(nn.Module):
    def __init__(self,in_filters,mid_filters,out_filters):
        super().__init__()

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_filters,mid_filters,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(mid_filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_filters,out_filters,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_filters,mid_filters,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(mid_filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_filters,out_filters,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_filters),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_filters,mid_filters,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(mid_filters),
            nn.LeakyReLU(0.1),
        )

    def forward(self,x):
        return self.conv5(x)


# Upsampling
class upsampling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsampling,self).__init__()

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2,mode='nearest')
        )
    def forward(self,x):
        return self.upsample(x)

# Downsampling
class downsampling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(downsampling,self).__init__()

        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self,x):
        return self.downsampling(x)

if __name__ == '__main__':
    x = torch.randn(5,3,416,416)
    backbone = CSPDarknet(3)
    
    x = backbone(x)
    
    # P1
    conv_x3 = conv_3(1024,512,1024)
    p1 = conv_x3(x[2])
    print(p1.shape) # batch_size x 512 x 13 x 13
    # SSP P2
    ssp = SSP()
    p2 = ssp(p1)
    print(p2.shape) # batch_size x 2048 x 13 x 13
    # P3
    conv_x3 = conv_3(2048,512,1024)
    p3 = conv_x3(p2)
    print(p3.shape) # batch_size x 512 x 13 x 13
    # P4
    upsample = upsampling(512,256)
    p4 = upsample(p3)
    print(p4.shape) # batch_size x 256 x 26 x 26

    # P5
    conv = conv_op(512,256,1,1)
    p5 = conv(x[1])
    print(p5.shape) # batch_size x 256 x 26 x 26

    # P6
    p6 = torch.cat((p4,p5),dim=1)
    conv_x5 = conv_5(512,256,512)
    p6 = conv_x5(p6)
    print(p6.shape)

    # P7
    upsample = upsampling(256,128)
    p7 = upsample(p6)
    print(p7.shape)

    # P8
    conv = conv_op(256,128,1,1)
    p8 = conv(x[0])
    print(p8.shape)

    # P9
    p9 = torch.cat((p7,p8),dim=1)
    conv_x5 = conv_5(256,128,256)
    p9 = conv_x5(p9)
    print(p9.shape)

    # P10
    downsample = downsampling(128,256)
    p10 = downsample(p9)
    print(p10.shape)