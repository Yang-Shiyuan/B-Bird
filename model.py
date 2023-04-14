
import torchvision
import torch, torch.nn as nn
from dataset import MyDataset
from utils import load_pickle, padded_cmap
import pandas as pd  #
from pathlib import Path
import timm

# class MyModel(nn.Module):
#     def __init__(self, num_classes=264):
#         super(MyModel, self).__init__()
#         self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)
#         self.classifier =nn.Sequential(
#             nn.Dropout(0.4),nn.Linear(1000, 1000),nn.ReLU(),
#             nn.Dropout(0.4),nn.Linear(1000, num_classes))
#
#     def forward(self, x):
#         x = self.mv2(x.repeat(1,3,1,1))  # [b, 3, f=12|128, t=4096]
#         x = self.classifier(x)
#         return x

class MyModel(nn.Module):
    def __init__(self, num_classes=264):
        super(MyModel, self).__init__()
        self.cnn = torchvision.models.efficientnet_b3(pretrained=True)
        self.classifier =nn.Sequential(
            nn.Dropout(0.4),nn.Linear(1000, 1000),nn.ReLU(),
            nn.Dropout(0.4),nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.cnn(x.repeat(1,3,1,1))  # [b, 3, f=12|128, t=4096]
        x = self.classifier(x)
        return x


class BaseModel(nn.Module):
    def __init__(self, num_classes=264):
        super(BaseModel, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b1_ns", pretrained=False)
        self.in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.in_features, num_classes)
        )

    def forward(self, x):
        logits = self.backbone(x.repeat(1,3,1,1))
        return logits