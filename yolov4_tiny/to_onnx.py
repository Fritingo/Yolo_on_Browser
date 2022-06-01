import torch

# from web_inf_arch import YoloBody
from inf_bbox import YoloBody
from test_model import test

anchors_mask = [[3,4,5], [1,2,3]]
classes_num = 20
phi = 2 

FloatTensor = torch.FloatTensor
grid_x = [torch.linspace(0, 13 - 1, 13).repeat(13, 1).type(FloatTensor),torch.linspace(0, 26 - 1, 26).repeat(26, 1).type(FloatTensor)]
grid_x = [torch.linspace(0, 13 - 1, 13).repeat(13, 1).repeat(3, 1, 1).tolist(),torch.linspace(0, 26 - 1, 26).repeat(26, 1).repeat(3, 1, 1).tolist()]
# print(grid_x)
   
def main():
  pytorch_model = YoloBody(anchors_mask,classes_num,grid_x,2)
  # pytorch_model = YoloBody(anchors_mask,classes_num,2)
  pytorch_model.load_state_dict(torch.load('best_epoch_weights2.pth'))
  pytorch_model.eval()
  print(pytorch_model)
  dummy_input = torch.zeros(1,3,416,416)
  # dummy_input = torch.zeros(416*416*4)
  torch.onnx.export(pytorch_model, dummy_input, 'yolov4_tiny.onnx', verbose=True,opset_version=10)


if __name__ == '__main__':
  main()