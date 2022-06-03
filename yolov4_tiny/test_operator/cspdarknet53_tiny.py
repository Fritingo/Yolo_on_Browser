import torch
import torch.nn as nn

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

# Residual Block (BC size same)
# -> BC ---> BC(out_filter/2)---> BC -> cat -> BC -> cat -> avgpool
#        |                    |          ^            ^                    
#        |                     ----------|            |                          
#        |                                            |                  
#         ---------------------------------------------
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResBlock,self).__init__()
        self.out_channels = out_channels
        
        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels, out_channels//2, 3)
        self.conv3 = BasicConv(out_channels//2, out_channels//2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.avgpool = nn.AvgPool2d([2,2],[2,2])
       

    def forward(self,x):
        x = self.conv1(x)
        route = x
        x = self.conv2(x)
        route1 = x
        x = self.conv3(x)
        x = torch.cat([x,route1], dim = 1) 
        x = self.conv4(x)
        x = torch.cat([route, x], dim = 1)
        x = self.avgpool(x)
        return x

# CSPDarknet53
# input 416x416x3
# -> BC -> BC -> ResBlock -> ResBlock -> ResBlock
#                               |            |
# output 
class CSPDarknet_tiny(nn.Module):
    def __init__(self,num_feature):
        super(CSPDarknet_tiny,self).__init__()

        self.num_feature = num_feature

        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
    
        self.stages = nn.ModuleList([
            # 104,104,64 -> 52,52,128
            ResBlock(64, 64),
            # 52,52,128 -> 26,26,256
            ResBlock(128, 128),
            # 26,26,256 -> 13,13,512
            ResBlock(256, 256),
            
        ])

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print('one---> ',x.shape)
        feature = []
        for stage in self.stages:
            x = stage(x)
            # print('stage---> ',x.shape)
            feature.append(x)

        return feature[-self.num_feature:]
        # return x
     
def cspdarknet53_tiny(pretrained, **kwargs):
    model = CSPDarknet_tiny(2)
    if pretrained:
        model.load_state_dict(torch.load("model_data/CSPdarknet53_tiny_backbone_weights.pth"),False)
        print('load weights')
    return model

if __name__ == '__main__':
    model = CSPDarknet_tiny(2)
    x = torch.randn(5,3,416,416)
    y = model(x)
    print(len(y))