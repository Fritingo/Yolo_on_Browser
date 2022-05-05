
from turtle import forward
from cv2 import pointPolygonTest
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


if __name__ == '__main__':
    x = torch.randn(1,3,416,416)
    backbone = CSPDarknet(3)
    
    x = backbone(x)
    
    # P1
    conv_x3 = conv_3(1024,512,1024)
    p1 = conv_x3(x[2])
    print(p1.shape)
    # SSP P2
    ssp = SSP()
    p2 = ssp(p1)
    print(p2.shape)
    # P3
    conv_x3 = conv_3(2048,512,1024)
    p3 = conv_x3(p2)
    print(p3.shape)
    # P4
    upsample = upsampling(512,256)
    p4 = upsample(p3)
    print(p4.shape)
