import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pretrainedmodels
from torch.nn import functional as F

# model
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

def run(fold):
    # train image path
    training_data_path = input("Enter the training path: ")
    # csv data path that was created from folds
    fold_csv_path = input("Enter the train_fold.csv file path: ")
    df = pd.read_csv(fold_csv_path)
    device = "cuda"
    epochs = 30
    train_batch_size = 32
    valid_batch_size = 16
    





