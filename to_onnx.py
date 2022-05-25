import torch

from yolov4 import yolo

anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
classes_num = 20

def main():
  pytorch_model = yolo(anchors_mask,classes_num)
  pytorch_model.load_state_dict(torch.load('ep200-2.pt'))
  pytorch_model.eval()
  dummy_input = torch.zeros(418*418*4)
  torch.onnx.export(pytorch_model, dummy_input, 'yolov4.onnx', verbose=True)


if __name__ == '__main__':
  main()