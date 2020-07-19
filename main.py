import os
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
import pretrainedmodels
from torch.nn import functional as F
import albumentations
from wtfml.data_loaders.image import ClassificationLoader
from apex import amp
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine


# model
class SEResNext(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext, self).__init__()
        self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](
            pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(batch_size, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))
        return out, loss 


def train(fold):
    # train image path
    training_data_path = input("Enter the training path: ")
    # csv data path that was created from folds
    fold_csv_path = input("Enter the train_fold.csv file path: ")
    df = pd.read_csv(fold_csv_path)
    model_path = "model/"
    device = "cuda"
    epochs = 30
    train_batch_size = 32
    valid_batch_size = 16
    mean = (0.485, 0.456, 0.225)
    standard_deviation = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # normalize images
    train_aug = albumentations.Compose([
        albumentations.Normalize(mean,
                                 std=standard_deviation,
                                 max_pixel_value=255.0,
                                 always_apply=True)
    ])

    valid_aug = albumentations.Compose([
        albumentations.Normalize(mean,
                                 std=standard_deviation,
                                 max_pixel_value=255.0,
                                 always_apply=True)
    ])

    # train image mapping
    train_images = df_train.image_name.values.tolist()
    train_images = [
        os.path.join(training_data_path, i + ".jpg") for i in train_images
    ]
    train_targets = df_train.target.values

    valid_images = df_train.image_name.values.tolist()
    valid_images = [
        os.path.join(training_data_path, i + ".jpg") for i in valid_images
    ]
    valid_targets = df_valid.target.values

    # create train loader
    train_dataset = ClassificationLoader(image_paths=train_images,
                                         targets=train_targets,
                                         resize=None,
                                         augmentations=train_aug)
    # train loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=4)

    # create valid dataset
    valid_dataset = ClassificationLoader(image_paths=valid_images,
                                         targets=valid_targets,
                                         resize=None,
                                         augmentations=valid_aug)
    # validation data loader
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=valid_batch_size,
                                               shuffle=False,
                                               num_workers=4)

    # import model
    model = SEResNext(pretrained='imagenet')
    model.to(device)
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # dynamic learning rate reducing based on validation measurements.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # https://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        optimizer,
        patience=4,
        mode='max',
    )
    # use apex for mixed precision training
    # amp: Automatic Mixed Precision
    model, optimizer = amp.initialize(model,
                                      optimizer,
                                      opt_level='01',
                                      verbosity=0)
    # earlystopping
    es = EarlyStopping(patience=5, mode='max')
    # train the train data
    # use thr wtfml module for calculating loss and evaluation
    for epoch in range(epochs):
        training_loss = Engine.train(train_loader,
                                     model,
                                     optimizer,
                                     device,
                                     fp16=True)
        predictions, valid_loss = Engine.evaluate(train_loader, model,
                                                  optimizer, device)
        predictions = np.vstack((predictions).ravel())
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"epoch = {epoch}, auc= {auc}")
        es(auc, model, model_path)

        if es.early_stop:
            print("Early Stopping")
            break


if __name__ == "__main__":
    train(fold=0)
