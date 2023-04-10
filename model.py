
import torchvision
import torch, torch.nn as nn
from dataset import MyDataset
from utils import load_pickle, padded_cmap
import pandas as pd  #
from pathlib import Path


class MyModel(nn.Module):
    def __init__(self, num_classes=264):
        super(MyModel, self).__init__()
        self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)
        self.classifier =nn.Sequential(
            nn.Dropout(0.4),nn.Linear(1000, 1000),nn.ReLU(),
            nn.Dropout(0.4),nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.mv2(x.repeat(1,3,1,1))  # [b, 3, f=12|128, t=4096]
        x = self.classifier(x)
        return x
