import onnxruntime
import numpy as np
import torch
import onnx
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from PIL import Image
from inf_bbox import YoloBody


ort_session = onnxruntime.InferenceSession("yolov4_tiny.onnx",providers=['CPUExecutionProvider'])


#-------------------------------
img = '/home/fritingo/Documents/pytorch_yolo/yolov4-tiny-pytorch/img/000122.jpg'
image = Image.open(img)
image_shape = np.array(np.shape(image)[0:2])
# print('image',image_shape)
#---------------------------------------------------------#
#   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
image       = cvtColor(image)
#---------------------------------------------------------#
#   给图像增加灰条，实现不失真的resize
#   也可以直接resize进行识别
#---------------------------------------------------------#
image_data  = resize_image(image, (416,416), False)
#---------------------------------------------------------#
#   添加上batch_size维度
#---------------------------------------------------------#
image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)


images = torch.from_numpy(image_data)
# print(images)
#-------------------------------


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

x = torch.randn(4*416*416)
# x = torch.linspace(0, 255,steps=4*416*416)
# print('hi',x.shape,x)
# compute ONNX Runtime output prediction
# t = ort_session.get_inputs()[0].name
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
ort_outs = ort_session.run(None, ort_inputs)

# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print(ort_outs)
# compare ONNX Runtime and PyTorch results
grid_x = [torch.linspace(0, 13 - 1, 13).repeat(13, 1).repeat(3, 1, 1).tolist(),torch.linspace(0, 26 - 1, 26).repeat(26, 1).repeat(3, 1, 1).tolist()]
# print(grid_x)
   

pytorch_model = YoloBody([[3,4,5], [1,2,3]],20,grid_x,2)


        
pytorch_model.load_state_dict(torch.load('best_epoch_weights2.pth', map_location='cpu'))
pytorch_model  = pytorch_model.eval()
torch_out = pytorch_model(images)
print(torch_out)
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")