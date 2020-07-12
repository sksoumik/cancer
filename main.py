import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pretrainedmodels
from torch.nn import functional as F


class SEResNext(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext, self).__init__()
        self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](
            pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image):
        batch_size, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(batch_size, -1)
        out = self.out(x)
        return out
