import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

from yolov4 import yolo

class BBox():
    def __init__(self,classes_num,inputs,anchors_mask):
        super(BBox,self).__init__()
        # yolo k means anchors
        self.anchors = [[12, 16],  [19, 36],  [40, 28],  [36, 75],  [76, 55],  [72, 146],  [142, 110],  [192, 243],  [459, 401]]
        self.classes_num = classes_num
        self.input_shape = inputs
        self.anchors_mask = anchors_mask

    def decode_box(self,inputs):
        outputs = []
        
        for i,input in enumerate(inputs):
            print(i,input.shape)



if __name__ == '__main__':
    x = torch.randn(5,3,416,416)

    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    classes_num = 20
    Yolo = yolo(anchors_mask,classes_num)

    o1,o2,o3 = Yolo(x)
    outputs = [o1,o2,o3]
    get_bbox = BBox(classes_num,outputs,anchors_mask)
    print(o1.shape,o2.shape,o3.shape)
    get_bbox.decode_box