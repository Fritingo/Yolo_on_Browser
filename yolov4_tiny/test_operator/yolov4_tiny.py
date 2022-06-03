from turtle import forward
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F

from cspdarknet53_tiny import cspdarknet53_tiny

# SPP

class SPP(nn.Module):
    def __init__(self,pool_sizes=[5,9,13]):
        super(SPP,self).__init__()

        self.avgpool0 = nn.AvgPool2d(pool_sizes[0],1,pool_sizes[0]//2)
        self.avgpool1 = nn.AvgPool2d(pool_sizes[1],1,pool_sizes[1]//2)
        self.avgpool2 = nn.AvgPool2d(pool_sizes[2],1,pool_sizes[2]//2)

    def forward(self,x):
        feature0 = self.avgpool0(x)
        # print('0',feature0.shape)
        feature1 = self.avgpool1(x)
        # print('1',feature1.shape)
        feature2 = self.avgpool2(x)
        # print('2',feature2.shape)
        # print('x',x.shape)

        features = torch.cat((feature0,feature1,feature2,x),dim=1)

        return features

# Basic Conv Block
# Conv -> BN -> Act
# args (in_channels, out_channels, kernel_size, stride)
class BasicConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1):
        super(BasicConv,self).__init__()
        # Conv in , out , kernel size , stride , padding , bias
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,kernel_size//2,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

# Upsample scale_factor = 2 nearest
class upsample(nn.Module):
    def __init__(self) -> None:
        super(upsample,self).__init__()
    
    def forward(self,x):
        b,c,h,w = x.size()
        target_h = 2 * h
        target_w = 2 * w

        x = x.view(b,c,h,1,w,1)
        x = x.expand(b,c,h,target_h//h,w,target_w//w).contiguous()
        x = x.view(b,c,target_h,target_w)
        
        return x


# ChannelAttention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            
            return self.sigmoid(avg_out)

# SpatialAttention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# CBAM
class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x

# yolo head
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m



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

# Yolo head
class yolo_head(nn.Module):
    def __init__(self,filters_list, in_filters):
        super(yolo_head,self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_filters,filters_list[0],kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(filters_list[0]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(filters_list[0],filters_list[1],kernel_size=1,stride=1,padding=0,bias=False),
        )
    
    def forward(self,x):
        return self.head(x)

# Yolo 
# backbone ---> cbam -> cat -> yolo_head 26x26
#           |            ^  
#           |        upsample
#           |            ^
#           |           BC
#           |            ^
#            -> cbam -> BC -> yolo_head 13x13
class yolo(nn.Module):
    def __init__(self,anchors_mask,classes_num):
        super(yolo,self).__init__()

        self.backbone = cspdarknet53_tiny(True)

        self.conv1 = BasicConv(512,256,1)

        self.yolo_h1 = yolo_head([512, len(anchors_mask[0]) * (5 + classes_num)],256)
       
        self.conv2 = BasicConv(256,128,1)
        self.upsample = upsample()

        self.yolo_h2 = yolo_head([256, len(anchors_mask[1]) * (5 + classes_num)],384)

        self.cbam1 = cbam_block(256)
        self.cbam2 = cbam_block(512)
        self.cbam_up = cbam_block(128)

    def forward(self,x):
        x = self.backbone(x)
        print('x',len(x),x[0].shape,x[1].shape)
        x1 = self.cbam1(x[0])
        print('x1',x1.shape)

        x2 = self.cbam2(x[1])
        print('x2',x2.shape)

        x2 = self.conv1(x2)
        print('x2_1',x2.shape)
        out1 = self.yolo_h1(x2)
        print('out1',out1.shape)

        x_upsample = self.conv2(x2)
        print('x_upsample',x_upsample.shape)
        x_upsample = self.upsample(x_upsample)
        print('x_upsample1',x_upsample.shape)
        x_upsample = self.cbam_up(x_upsample)
        print('x_upsample2',x_upsample.shape)
        x1 = torch.cat([x1,x_upsample],axis=1)
        print('x1',x1.shape)
        out2 = self.yolo_h2(x1)
        print('out2',out2.shape)

        return out1,out2


if __name__ == '__main__':
    x = torch.randn(5,3,416,416)

    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    classes_num = 20
    Yolo = yolo(anchors_mask,classes_num)

    o1,o2 = Yolo(x)
    print(o1.shape)
    print(o2.shape)
    

    # --------------- test shape ----------------
    # backbone = CSPDarknet(3)
    
    # x = backbone(x)
    
    # # P1
    # conv_x3 = conv_3(1024,512,1024)
    # p1 = conv_x3(x[2])
    # print(p1.shape) # batch_size x 512 x 13 x 13
    # # SSP P2
    # ssp = SSP()
    # p2 = ssp(p1)
    # print(p2.shape) # batch_size x 2048 x 13 x 13
    # # P3
    # conv_x3 = conv_3(2048,512,1024)
    # p3 = conv_x3(p2)
    # print(p3.shape) # batch_size x 512 x 13 x 13
    # # P4
    # upsample = upsampling(512,256)
    # p4 = upsample(p3)
    # print(p4.shape) # batch_size x 256 x 26 x 26

    # # P5
    # conv = conv_op(512,256,1,1)
    # p5 = conv(x[1])
    # print(p5.shape) # batch_size x 256 x 26 x 26

    # # P6
    # p6 = torch.cat((p4,p5),dim=1)
    # conv_x5 = conv_5(512,256,512)
    # p6 = conv_x5(p6)
    # print(p6.shape)

    # # P7
    # upsample = upsampling(256,128)
    # p7 = upsample(p6)
    # print(p7.shape)

    # # P8
    # conv = conv_op(256,128,1,1)
    # p8 = conv(x[0])
    # print(p8.shape)

    # # P9
    # p9 = torch.cat((p7,p8),dim=1)
    # conv_x5 = conv_5(256,128,256)
    # p9 = conv_x5(p9)
    # print(p9.shape)

    # # H1
    # class_num = 20
    # yolohead1 = yolo_head(128,256,3*(5+class_num))
    # out1 = yolohead1(p9)
    # print('h1',out1.shape)

    # # P10
    # downsample = downsampling(128,256)
    # p10 = downsample(p9)
    # print(p10.shape)

    # # P11
    # p11 = torch.cat((p6,p10),dim=1)
    # conv_x5 = conv_5(512,256,512)
    # p11 = conv_x5(p11)
    # print(p11.shape)

    # # H2
    # yolohead2 = yolo_head(256,512,3*(5+class_num))
    # out2 = yolohead2(p11)
    # print('h2',out2.shape)

    # # P12
    # downsample = downsampling(256,512)
    # p12 = downsample(p11)
    # print(p12.shape)

    # # P13
    # p13 = torch.cat((p3,p12),dim=1)
    # conv_x5 = conv_5(1024,512,1024)
    # p13 = conv_x5(p13)
    # print(p13.shape)

    # # H3
    # yolohead3 = yolo_head(512,1024,3*(5+class_num))
    # out3 = yolohead3(p13)
    # print('h3',out3.shape)