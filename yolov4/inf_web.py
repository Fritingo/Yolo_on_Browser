import torch
import torch.nn as nn
from torchvision.ops import nms

from nets.CSPdarknet53_tiny import darknet53_tiny
from nets.attention import cbam_block, eca_block, se_block, CA_Block

from utils.utils import get_anchors

attention_block = [se_block, cbam_block, eca_block, CA_Block]

anchors_path      = 'model_data/yolo_anchors.txt'
anchors, num_anchors      = get_anchors(anchors_path)

class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[3,4,5], [0,1,2]]):
        super(DecodeBox, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        
        self.anchors_mask   = anchors_mask

    def decode_box(self, inputs,grid_x):
        outputs = []
        for i, input in enumerate(inputs):
            
            batch_size      = input.size(0)
            input_height    = input.size(2)
            input_width     = input.size(3)

           
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
           
            # scaled_anchors = []
            # for anchor_index in self.anchors_mask[i]:
            #     # print(anchor_index,self.anchors)
            #     anchor_width, anchor_height = self.anchors[anchor_index]
            #     # print(anchor_width,anchor_height)
            #     scaled_anchors.append([anchor_width/stride_w,anchor_height/stride_h])
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

           
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            #-----------ok

            
            x = torch.sigmoid(prediction[..., 0])  
            y = torch.sigmoid(prediction[..., 1])
           
            w = prediction[..., 2]
            h = prediction[..., 3]
           
            conf        = torch.sigmoid(prediction[..., 4])
            
            pred_cls    = torch.sigmoid(prediction[..., 5:])

           
            FloatTensor = torch.FloatTensor
            LongTensor  = torch.LongTensor

            
            p_grid_x = grid_x[i]
            print(p_grid_x.shape,x.shape,y.shape)
            p_grid_x = p_grid_x.view(x.shape).type(FloatTensor)
            p_grid_y = p_grid_x.view(y.shape).type(FloatTensor)
           
            anchor_w_b = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h_b = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

            
            anchor_w_b = anchor_w_b.view(1,3,1,1)
            anchor_h_b = anchor_h_b.view(1,3,1,1)
            
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
            
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
            
        return outputs

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
       
        print('hey',type(prediction),type(prediction[0][0]))
        box_corner          = prediction.new(prediction.shape)
        print('bbox',box_corner.shape)
        box_corner0 = box_corner[:, :, 0].clone()
        box_corner0 = box_corner0 - prediction[:, :, 2] / 2
        box_corner0 = box_corner0.view(1,2535,1)
        box_corner1 = box_corner[:, :, 1].clone()
        box_corner1 = box_corner1 - prediction[:, :, 3] / 2
        box_corner1 = box_corner1.view(1,2535,1)
        box_corner2 = box_corner[:, :, 0].clone()
        box_corner2 = box_corner2 + prediction[:, :, 2] / 2
        box_corner2 = box_corner2.view(1,2535,1)
        box_corner3 = box_corner[:, :, 1].clone()
        box_corner3 = box_corner3 + prediction[:, :, 3] / 2
        box_corner3 = box_corner3.view(1,2535,1)
        print(box_corner0.shape)
        # box_corner = torch.cat([box_corner0,box_corner1],axis=2)

        box_corner = torch.cat([box_corner0,box_corner1,box_corner2,box_corner3],axis=2)

        print('bbox2',box_corner.shape)

        prediction = prediction[:, :, 4:]
        prediction = torch.cat([box_corner,prediction],axis=2)
        print(prediction.shape)

 

       

        output = [None for _ in range(len(prediction))]

        
        max_detections = 0
        for i, image_pred in enumerate(prediction):

            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

         
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

       
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            print('there',image_pred.size(0))
            if image_pred.size(0) == 0:
                return -1
            
            detections = torch.cat([image_pred[:, :5], class_conf.float(), class_pred.float()], 1).type(torch.FloatTensor)
            
            unique_labels = detections[:, -1].cpu().unique()

         

            for c in unique_labels:
                
                detections_class = detections[detections[:, -1] == c]

            
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]
                
           
                output = max_detections
        output = torch.Tensor(output).type(torch.FloatTensor)
        return box_corner


#-------------------------------------------------#
#   ????????? -> ?????? + ????????? + ????????????
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
#   ?????? + ?????????
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
    def __init__(self, anchors_mask, num_classes, grid_x, phi=0, pretrained=False):
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

        if 1 <= self.phi and self.phi <= 4:
            self.feat1_att      = attention_block[self.phi - 1](256)
            self.feat2_att      = attention_block[self.phi - 1](512)
            self.upsample_att   = attention_block[self.phi - 1](128)

        #--------------bbox-----------------
        self.bbox = DecodeBox(anchors, num_classes, (416, 416), anchors_mask)

    def forward(self, x):
        #---------------------------------------------------# web
        x = x.reshape(416, 416, 4)
        x = x[:,:,:3]
        
        x = x.permute(2,0,1)
        
        x = x.reshape(-1,3,416,416)

        x = x / 255

        #---------------------------------------------------#
        #   ??????CSPdarknet53_tiny???????????????
        #   feat1???shape???26,26,256
        #   feat2???shape???13,13,512
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
        bboxes = self.bbox.decode_box([out0,out1],self.grid_x)

        print('yee',type(bboxes),len(bboxes),bboxes[0].shape)
        bboxes = torch.cat(bboxes,axis=1)
        # ---------------is tensor---------------
        # bboxes = self.bbox.non_max_suppression(torch.cat(bboxes, 1), 20, [416,416], 
        #                 [416,416], False, conf_thres = 0.5, nms_thres = 0.3)
        
        # print('all',bboxes,type(bboxes))
        # for i in range(bboxes.shape[0]):
        #     print(i,bboxes[i],type(bboxes[i]))
        bboxes = bboxes
        ans = bboxes
        # print(len(bboxes),bboxes[0].shape,bboxes[1].shape)
        return ans

if __name__ == '__main__':
    x = torch.randn(4*416*416)

    anchors_mask = [[3, 4, 5], [0, 1, 2]]
    classes_num = 20
    grid_x = [torch.linspace(0, 13 - 1, 13).repeat(13, 1).repeat(3, 1, 1),torch.linspace(0, 26 - 1, 26).repeat(26, 1).repeat(3, 1, 1)]

    Yolo = YoloBody(anchors_mask,classes_num,grid_x,2)

    o1 = Yolo(x)
    print(o1,type(o1))
    
    # .repeat(batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
    # print(grid_x)
    # print(o2.shape)
    