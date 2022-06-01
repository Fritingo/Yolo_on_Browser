from matplotlib.pyplot import axis
import torch
import torch.nn as nn

from nets.CSPdarknet53_tiny import darknet53_tiny
from nets.attention import cbam_block, eca_block, se_block, CA_Block

from utils.utils import get_anchors

attention_block = [se_block, cbam_block, eca_block, CA_Block]

anchors_path      = 'model_data/yolo_anchors.txt'
anchors, num_anchors      = get_anchors(anchors_path)
from torchvision.ops import nms

class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[3,4,5], [0,1,2]]):
        super(DecodeBox, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[81,82],[135,169],[344,319]
        #   26x26的特征层对应的anchor是[10,14],[23,27],[37,58]
        #-----------------------------------------------------------#
        self.anchors_mask   = anchors_mask

    def decode_box(self, inputs,grid_x):
        outputs = []
        for i, input in enumerate(inputs):
            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 255, 13, 13
            #   batch_size, 255, 26, 26
            #-----------------------------------------------#
            batch_size      = input.size(0)
            input_height    = input.size(2)
            input_width     = input.size(3)
            print(input_height)
            #-----------------------------------------------#
            #   输入为416x416时
            #   stride_h = stride_w = 32、16、8
            #-----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            # #-------------------------------------------------#
            # #   此时获得的scaled_anchors大小是相对于特征层的
            # #-------------------------------------------------#
            # # print(self.anchors)
            scaled_anchors_tensor = torch.FloatTensor([])
            for item_i ,anchor_index in enumerate(self.anchors_mask[i]):
                anchor_width, anchor_height = self.anchors[anchor_index]
                if item_i == 0:
                    scaled_anchors_tensor = torch.FloatTensor([anchor_width/stride_w,anchor_height/stride_h])
                    # scaled_anchors_tensor[0] = anchor_width/stride_w
                    # scaled_anchors_tensor[1] = anchor_height/stride_h
                    scaled_anchors_tensor = scaled_anchors_tensor.view(1,2)
                else:
                    scaled_temp = torch.FloatTensor([anchor_width/stride_w,anchor_height/stride_h])
                    scaled_temp = scaled_temp.view(1,2)
                    scaled_anchors_tensor = torch.cat([scaled_anchors_tensor,scaled_temp],axis=0)
            print('pp',scaled_anchors_tensor.shape,scaled_anchors_tensor)

            # scaled_anchors = []
            # for anchor_index in self.anchors_mask[i]:
            #     # print(anchor_index,self.anchors)
            #     anchor_width, anchor_height = self.anchors[anchor_index]
            #     # print(anchor_width,anchor_height)
            #     scaled_anchors.append([anchor_width/stride_w,anchor_height/stride_h])
            # # scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]
            # print('qq',len(scaled_anchors),scaled_anchors)
            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 13, 13, 85
            #   batch_size, 3, 26, 26, 85
            #-----------------------------------------------#
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            #-----------ok

            #-----------------------------------------------#
            #   先验框的中心位置的调整参数
            #-----------------------------------------------#
            x = torch.sigmoid(prediction[..., 0])  
            print('prediction',prediction.shape,'x',x.shape)
            y = torch.sigmoid(prediction[..., 1])
            # #-----------------------------------------------#
            # #   先验框的宽高调整参数
            # #-----------------------------------------------#
            w = prediction[..., 2]
            h = prediction[..., 3]
            # #-----------------------------------------------#
            # #   获得置信度，是否有物体
            # #-----------------------------------------------#
            conf        = torch.sigmoid(prediction[..., 4])
            # #-----------------------------------------------#
            # #   种类置信度
            # #-----------------------------------------------#
            pred_cls    = torch.sigmoid(prediction[..., 5:])

            # FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            # LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            FloatTensor = torch.FloatTensor
            LongTensor  = torch.LongTensor

            #----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角 
            #   batch_size,3,13,13
            #----------------------------------------------------------#
            # test = torch.linspace(0, input_width - 1, input_width)
            # print(batch_size,len(self.anchors_mask[i]))
            p_grid_x = grid_x[i]
            p_grid_x = torch.FloatTensor(p_grid_x)
            p_grid_y = p_grid_x.view(y.shape).type(FloatTensor)
            p_grid_x = p_grid_x.view(x.shape).type(FloatTensor)
            
            # grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            #     batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            # grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            #     batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

        #     #----------------------------------------------------------#
        #     #   按照网格格式生成先验框的宽高
        #     #   batch_size,3,13,13
        #     #----------------------------------------------------------#
            anchor_w_b = scaled_anchors_tensor.index_select(1, LongTensor([0]))
            anchor_h_b = scaled_anchors_tensor.index_select(1, LongTensor([1]))

            # anchor_wf = anchor_w_b.repeat(1, 1, input_height * input_width).view(w.shape)
            # print('orignal',anchor_wf,anchor_wf.shape)
            # print('1',anchor_w_b,anchor_w_b.shape)
            anchor_w_b = anchor_w_b.view(1,3,1,1)
            anchor_h_b = anchor_h_b.view(1,3,1,1)
            # print('1-2',anchor_w_b,anchor_w_b.shape)
            anchor_w_b1 = anchor_w_b.clone()
            anchor_h_b1 = anchor_h_b.clone()

            for _ in range(input_height-1):
                anchor_w_b1 = torch.cat([anchor_w_b1,anchor_w_b],axis=2)
                anchor_h_b1 = torch.cat([anchor_h_b1,anchor_h_b],axis=2)
                

            anchor_w = anchor_w_b1.clone()
            anchor_h = anchor_h_b1.clone()
            for _ in range(input_width-1):
                anchor_w = torch.cat([anchor_w,anchor_w_b1],axis=3)
                anchor_h = torch.cat([anchor_h,anchor_h_b1],axis=3)
            
            
            # print('2',anchor_w,anchor_w.shape)
            # anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            # anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        #     #----------------------------------------------------------#
        #     #   利用预测结果对先验框进行调整
        #     #   首先调整先验框的中心，从先验框中心向右下角偏移
        #     #   再调整先验框的宽高。
        #     #----------------------------------------------------------#
        #-------------- not sure-------------------
            pred_boxes = x.data + p_grid_x
            # print(pred_boxes.shape)
            pred_boxes = pred_boxes.reshape(1,3,input_height,input_width,1)
            pred_boxes_b = y.data + p_grid_y
            pred_boxes_b = pred_boxes_b.reshape(1,3,input_height,input_width,1)
            pred_boxes = torch.cat([pred_boxes,pred_boxes_b],axis=4)
            pred_boxes_b = torch.exp(w.data) * anchor_w
            pred_boxes_b = pred_boxes_b.reshape(1,3,input_height,input_width,1)
            pred_boxes = torch.cat([pred_boxes,pred_boxes_b],axis=4)
            pred_boxes_b = torch.exp(h.data) * anchor_h
            pred_boxes_b = pred_boxes_b.reshape(1,3,input_height,input_width,1)
            pred_boxes = torch.cat([pred_boxes,pred_boxes_b],axis=4)
            pred_boxes = pred_boxes.type(FloatTensor)
            # print(pred_boxes.shape)
            # pred_boxes          = FloatTensor(prediction[..., :4].shape)
            
            
            # pred_boxes[..., 0]  = x.data + p_grid_x
            # print('orignal',pred_boxes,pred_boxes.shape)
            # pred_boxes[..., 1]  = y.data + p_grid_y
            # pred_boxes[..., 2]  = torch.exp(w.data) * anchor_w
            # pred_boxes[..., 3]  = torch.exp(h.data) * anchor_h
            # print(pred_boxes.shape)

        #     #----------------------------------------------------------#
        #     #   将输出结果归一化成小数的形式
        #     #----------------------------------------------------------#

            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            # print('ooo',type(output),output.shape,type(output[0][0][0]))
            # if i == 0:
            #     outputs = output.data
            #     print('here',type(outputs))
            #     outputs = outputs.view(1,output.shape[0],output.shape[1],output.shape[2])
            # else:
            #     output = output.view(1,output.shape[0],output.shape[1],output.shape[2])
            #     outputs = torch.cat([outputs,output],axis=0)
            outputs.append(output.data)
        
        
            
        return pred_boxes


        # pred_boxes = pred_boxes.clone()
        # return _scale


    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        # input_shape = np.array(input_shape)
        # image_shape = np.array(image_shape)

        # if letterbox_image:
        #     #-----------------------------------------------------------------#
        #     #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #     #   new_shape指的是宽高缩放情况
        #     #-----------------------------------------------------------------#
        #     new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        #     offset  = (input_shape - new_shape)/2./input_shape
        #     scale   = input_shape/new_shape

        #     box_yx  = (box_yx - offset) * scale
        #     box_hw *= scale

        # box_mins    = box_yx - (box_hw / 2.)
        # box_maxes   = box_yx + (box_hw / 2.)
        # boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        # boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        # return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        #----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #----------------------------------------------------------#
    #     print('hey',type(prediction),type(prediction[0][0]))
        box_corner          = prediction.new(prediction.shape)
        print('bbox',type(box_corner))
        box_corner0 = box_corner[:, :, 0].clone()
    #     box_corner0 = box_corner0 - prediction[:, :, 2] / 2
    #     box_corner0 = box_corner0.view(1,2535,1)
    #     box_corner1 = box_corner[:, :, 1].clone()
    #     box_corner1 = box_corner1 - prediction[:, :, 3] / 2
    #     box_corner1 = box_corner1.view(1,2535,1)
    #     box_corner2 = box_corner[:, :, 0].clone()
    #     box_corner2 = box_corner2 + prediction[:, :, 2] / 2
    #     box_corner2 = box_corner2.view(1,2535,1)
    #     box_corner3 = box_corner[:, :, 1].clone()
    #     box_corner3 = box_corner3 + prediction[:, :, 3] / 2
    #     box_corner3 = box_corner3.view(1,2535,1)
    #     print(box_corner0.shape)
    #     # box_corner = torch.cat([box_corner0,box_corner1],axis=2)

    #     box_corner = torch.cat([box_corner0,box_corner1,box_corner2,box_corner3],axis=2)

    #     print('bbox2',box_corner.shape)

    #     prediction = prediction[:, :, 4:]
    #     prediction = torch.cat([box_corner,prediction],axis=2)
    #     print(prediction.shape)

    #     # box_corner          = prediction.new(prediction.shape)
    #     # box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    #     # box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    #     # box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    #     # box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    #     # prediction[:, :, :4] = box_corner[:, :, :4]

       

    #     output = [None for _ in range(len(prediction))]

        
    #     max_detections = 0
    #     for i, image_pred in enumerate(prediction):
    #         #----------------------------------------------------------#
    #         #   对种类预测部分取max。
    #         #   class_conf  [num_anchors, 1]    种类置信度
    #         #   class_pred  [num_anchors, 1]    种类
    #         #----------------------------------------------------------#
    #         class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

    #         # #----------------------------------------------------------#
    #         # #   利用置信度进行第一轮筛选
    #         # #----------------------------------------------------------#
    #         conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

    #         # #----------------------------------------------------------#
    #         # #   根据置信度进行预测结果的筛选
    #         # #----------------------------------------------------------#
    #         image_pred = image_pred[conf_mask]
    #         class_conf = class_conf[conf_mask]
    #         class_pred = class_pred[conf_mask]
    #         print('there',image_pred.size(0))
    #         if image_pred.size(0) == 0:
    #             return -1
    #         # #-------------------------------------------------------------------------#
    #         # #   detections  [num_anchors, 7]
    #         # #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
    #         # #-------------------------------------------------------------------------#
    #         detections = torch.cat([image_pred[:, :5], class_conf.float(), class_pred.float()], 1).type(torch.FloatTensor)
    #         # print(detections.shape,detections)
    #         # detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
    #         # print(detections.shape,detections)
    #         # #------------------------------------------#
    #         # #   获得预测结果中包含的所有种类
    #         # #------------------------------------------#
    #         unique_labels = detections[:, -1].cpu().unique()

    #         # if prediction.is_cuda:
    #         #     unique_labels = unique_labels.cuda()
    #         #     detections = detections.cuda()

    #         for c in unique_labels:
    #             #------------------------------------------#
    #             #   获得某一类得分筛选后全部的预测结果
    #             #------------------------------------------#
    #             detections_class = detections[detections[:, -1] == c]

    #         #     #------------------------------------------#
    #         #     #   使用官方自带的非极大抑制会速度更快一些！
    #         #     #------------------------------------------#
    #             keep = nms(
    #                 detections_class[:, :4],
    #                 detections_class[:, 4] * detections_class[:, 5],
    #                 nms_thres
    #             )
    #             max_detections = detections_class[keep]
    #  #---------------------------------------------
            #     # # 按照存在物体的置信度排序
            #     _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            #     detections_class = detections_class[conf_sort_index]
            # #     # # 进行非极大抑制
            #     max_detections = []
            #     while detections_class.size(0):
            #         # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #         max_detections.append(detections_class[0].unsqueeze(0))
            #         if len(detections_class) == 1:
            #             break
            #         ious = bbox_iou(max_detections[-1], detections_class[1:])
            #         detections_class = detections_class[1:][ious < nms_thres]
            #     # 堆叠
            #     max_detections = torch.cat(max_detections).data
                
            #     # Add max detections to outputs
            # print('hi',max_detections.shape)
            # if output[i] is not None:
            #     if i == 0:
            #         output = max_detections

                # else:
                #     output = 
                # output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            #     print(output)
            # if output[i] is not None:
            #     # output[i]           = output[i].cpu().numpy()
            #     box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            #     print(box_xy)
                # output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        #---------------------------------------------
        #         output = max_detections
        # output = torch.Tensor(output).type(torch.FloatTensor)
        return box_corner0








#-------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes,grid_x, phi=0, pretrained=False):
        super(YoloBody, self).__init__()
        self.phi            = phi
        self.backbone       = darknet53_tiny(pretrained)

        self.conv_for_P5    = BasicConv(512,256,1)
        self.yolo_headP5    = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)

        self.upsample       = Upsample(256,128)
        self.yolo_headP4    = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],384)

        self.conv_for_test    = BasicConv(256,128,1)
        self.yolo_head_test = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],128)

        self.grid_x = grid_x
        self.num_classes = num_classes
        self.flatten = nn.Flatten()

        if 1 <= self.phi and self.phi <= 4:
            self.feat1_att      = attention_block[self.phi - 1](256)
            self.feat2_att      = attention_block[self.phi - 1](512)
            self.upsample_att   = attention_block[self.phi - 1](128)

        #--------------bbox-----------------
        self.bbox = DecodeBox(anchors, num_classes, (416, 416), anchors_mask)

    def forward(self, x):
        #---------------------------------------------------# web
        # x = x.reshape(416, 416, 4)
        # x = x[:,:,:3]
        
        # x = x.permute(2,0,1)
        
        # x = x.reshape(-1,3,416,416)

        # x = x / 255

        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        #---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        if 1 <= self.phi and self.phi <= 4:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)

        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        out0 = self.yolo_headP5(P5) 

       
        #------------------------
        P4 = self.conv_for_test(feat1)
        # print(P4.shape)
        out1 = self.yolo_head_test(P4)
        #--------------------------

        #----bbox------------------
        # bboxes = self.bbox.decode_box([out0,out1],self.grid_x)

        # print('yee',type(bboxes),len(bboxes),bboxes[0].shape)
        # bboxes = torch.cat(bboxes,axis=1)
        # ---------------is tensor---------------
        # bboxes = self.bbox.non_max_suppression(bboxes, 20, [416,416], 
        #                 [416,416], False, conf_thres = 0.5, nms_thres = 0.3)
        
        # # print('all',bboxes,type(bboxes))
        # # for i in range(bboxes.shape[0]):
        # #     print(i,bboxes[i],type(bboxes[i]))
        # bboxes = bboxes
        # ans = bboxes[0]
        # print(ans.shape)
        # print(len(bboxes),bboxes[0].shape,bboxes[1].shape)
        # bboxes = self.flatten(bboxes)
        # print(bboxes)
        # bboxes[0,0,0,0,0] = 0
        # print(bboxes.shape)
        # print(out0[0][0].shape) 
        # conf = out0[0][4]
        # out = torch.cat()
        # out0 = out0.view(1, 3, 25, 13, 13).permute(0, 1, 3, 4, 2).contiguous()
        # print(out0.shape)
        # out = torch.sigmoid(out0)
        return out0

if __name__ == '__main__':
    x = torch.randn(4*416*416)

    anchors_mask = [[3, 4, 5], [0, 1, 2]]
    classes_num = 20
    grid_x = [torch.linspace(0, 13 - 1, 13).repeat(13, 1).repeat(3, 1, 1).tolist(),torch.linspace(0, 26 - 1, 26).repeat(26, 1).repeat(3, 1, 1).tolist()]

    Yolo = YoloBody(anchors_mask,classes_num,grid_x,2)

    o1,o2 = Yolo(x)
    print(o1,type(o1))
    
    # .repeat(batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
    # print(grid_x)
    # print(o2.shape)
    