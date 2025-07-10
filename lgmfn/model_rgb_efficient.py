import torch
import torch.nn as nn
from efficientnet import EfficientNet


class Model(nn.Module):
    def __init__(self, num_class=60, weights_path = '/weights/efficientnet-b7-dcc49843.pth'):
        super().__init__()
        
        self.efficientnet = EfficientNet.from_pretrained(model_name = 'efficientnet-b7', weights_path=weights_path, num_classes = num_class)
        


    def forward(self, x_rgb):
        out = self.efficientnet.forward(x_rgb)
        

        return out

