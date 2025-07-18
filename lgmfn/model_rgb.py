import torch
import torch.nn as nn
from resnet import resnet18 as ResNet


class Model(nn.Module):
    def __init__(self, num_class=60):
        super().__init__()
        self.resnet = ResNet(pretrained=True)
        numFit = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(numFit, num_class)  # resnet18
        # self.resnet.fc = nn.Linear(512, num_class) # resnet18


    def forward(self, x_rgb):
        rgb_weighted = x_rgb
        x = self.resnet.conv1(rgb_weighted)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.resnet.fc(x)
        # out = self.resnet(x_rgb)

        return out

