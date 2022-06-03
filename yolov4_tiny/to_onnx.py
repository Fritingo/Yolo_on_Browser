import torch

from web_yolo import yolo


anchors_mask = [[3,4,5], [1,2,3]]
classes_num = 19

def main():
  pytorch_model = yolo(anchors_mask,classes_num)
  # pytorch_model = YoloBody(anchors_mask,classes_num,2)
  pytorch_model.load_state_dict(torch.load('animal.pth'))
  pytorch_model.eval()
  print(pytorch_model)
  # dummy_input = torch.zeros(1,3,416,416)
  dummy_input = torch.zeros(416*416*4)
  torch.onnx.export(pytorch_model, dummy_input, 'yolov4_tiny.onnx', verbose=True,opset_version=10)


if __name__ == '__main__':
  main()