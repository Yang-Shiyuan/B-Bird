# import os
import gc
import timm
import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts #, ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint #, EarlyStopping,BackboneFinetuning

# import wandb
import albumentations as A
from torchtoolbox.tools import mixup_data, mixup_criterion
import soundfile as sf

import warnings
warnings.filterwarnings('ignore')


class Config:
    use_aug = False
    num_classes = 264
    batch_size = 64
    epochs = 50  # 12, 50
    PRECISION = 16
    PATIENCE = 8
    seed = 2023
    model = "tf_efficientnet_b2_ns"
    pretrained = True
    weight_decay = 1e-3
    use_mixup = True
    mixup_alpha = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = "./"
    train_images = "./specs/train/"
    valid_images = "./specs/valid/"
    train_path = "./train.csv"
    valid_path = "./valid.csv"

    SR = 32000
    DURATION = 5
    MAX_READ_SAMPLES = 5
    LR = 5e-4
    save_path = "./exp1/"
    bird_name_path = '/mnt/lyh/Code/bird_names.pickle3'
    pickle_file_path = '/mnt/lyh/Code/train_mel(dB,sr=32k,bin=128).pickle3'

pl.seed_everything(Config.seed, workers=True)

import pickle
def load_pickle(fname):
    f = open(fname, 'rb')
    out = pickle.load(f)
    f.close()
    return out

def config_to_dict(cfg):
    return dict((name, getattr(cfg, name)) for name in dir(cfg) if not name.startswith('__'))


df_train = pd.read_csv(Config.train_path)
df_valid = pd.read_csv(Config.valid_path)
df_train.head()

Config.num_classes = len(df_train.primary_label.unique())
print(Config.num_classes, type(df_train))

df_train = pd.concat([df_train, pd.get_dummies(df_train['primary_label'])], axis=1)
df_valid = pd.concat([df_valid, pd.get_dummies(df_valid['primary_label'])], axis=1)

# convert the class name into one-hot encoding
df_train.head()

birds = list(df_train.primary_label.unique())
print(len(birds))

missing_birds = list(set(list(df_train.primary_label.unique())).difference(list(df_valid.primary_label.unique())))
non_missing_birds = list(set(list(df_train.primary_label.unique())).difference(missing_birds))
print(len(missing_birds), len(non_missing_birds))

df_valid[missing_birds] = 0.0

# print(df_valid.primary_label)
# print(df_valid.iloc[:,17:])
# print('---------------------')
# print(df_train.columns)
df_valid = df_valid[df_train.columns] ## Fix order
# print(df_valid.iloc[:,17:])

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([A.Cutout(max_h_size=5, max_w_size=16), A.CoarseDropout(max_holes=4),], p=0.5),
        ])

class BirdDataset(Dataset):
    def __init__(self, df, sr = Config.SR, duration = Config.DURATION, augmentations = None, train = True):
        self.df = df
        self.sr = sr
        self.train = train
        self.duration = duration
        self.augmentations = augmentations
        if train:
            self.img_dir = Config.train_images
        else:
            self.img_dir = Config.valid_images

    def __len__(self):
        return len(self.df)

    @staticmethod
    def normalize(image):
        image = image / 255.0
        #image = torch.stack([image, image, image])
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        impath = self.img_dir + f"{row.filename}.npy"
        image = np.load(str(impath))[:Config.MAX_READ_SAMPLES]
        # print(type(image), image.shape, len(image))
        ########## RANDOM SAMPLING ################
        if self.train:
            image = image[np.random.choice(len(image))]
        else:
            image = image[0]
        #####################################################################
        image = torch.tensor(image).float()
        if self.augmentations:
            image = self.augmentations(image.unsqueeze(0)).squeeze()
        image.size()
        image = torch.stack([image, image, image])
        image = self.normalize(image)
        return image, torch.tensor(row[17:]).float()



def get_fold_dls(df_train, df_valid):
    ds_train = BirdDataset(
        df_train,
        sr = Config.SR,
        duration = Config.DURATION,
        augmentations = None,
        train = True
    )
    ds_val = BirdDataset(
        df_valid,
        sr = Config.SR,
        duration = Config.DURATION,
        augmentations = None,
        train = False
    )
    dl_train = DataLoader(ds_train, batch_size=Config.batch_size , shuffle=True) #, num_workers = 0)
    dl_val = DataLoader(ds_val, batch_size=Config.batch_size) #, num_workers = 0)
    return dl_train, dl_val, ds_train, ds_val


dl_train, dl_val, ds_train, ds_val = get_fold_dls(df_train, df_valid)


def get_optimizer(lr, params):
    model_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, params),
        lr=lr,
        weight_decay=Config.weight_decay
    )
    interval = "epoch"

    lr_scheduler = CosineAnnealingWarmRestarts(
        model_optimizer,
        T_0=Config.epochs,
        T_mult=1,
        eta_min=1e-6,
        last_epoch=-1
    )

    return {
        "optimizer": model_optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": interval,
            "monitor": "val_loss",
            "frequency": 1
        }
    }


class BirdClefModel(pl.LightningModule):
    def __init__(self, model_name=Config.model, num_classes=Config.num_classes, pretrained=Config.pretrained):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if 'res' in model_name:
            self.in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(self.in_features, num_classes)
        elif 'dense' in model_name:
            self.in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(self.in_features, num_classes)
        elif 'efficientnet' in model_name:
            self.in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(self.in_features, num_classes)
            )

        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, images):
        logits = self.backbone(images)
        return logits

    def configure_optimizers(self):
        return get_optimizer(lr=Config.LR, params=self.parameters())

    def train_with_mixup(self, X, y):
        X, y_a, y_b, lam = mixup_data(X, y, alpha=Config.mixup_alpha)
        y_pred = self(X)
        loss_mixup = mixup_criterion(F.cross_entropy, y_pred, y_a, y_b, lam)
        return loss_mixup

    def training_step(self, batch, batch_idx):
        image, target = batch
        if Config.use_mixup:
            loss = self.train_with_mixup(image, target)
        else:
            y_pred = self(image)
            loss = self.loss_function(y_pred, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        y_pred = self(image)
        val_loss = self.loss_function(y_pred, target)

        self.validation_step_outputs = {"val_loss": val_loss, "logits": y_pred, "targets": target}
        self.log("val_loss", val_loss)

        return self.validation_step_outputs

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader

    def on_validation_epoch_end(self):

        avg_loss = self.validation_step_outputs['val_loss'].mean()
        output_val = self.validation_step_outputs['logits'].sigmoid().cpu().detach().numpy()
        target_val = self.validation_step_outputs['targets'].cpu().detach().numpy()

        avg_score = sklearn.metrics.label_ranking_average_precision_score(target_val, output_val)

        self.log('val_accuracy', avg_score)
        self.validation_step_outputs.clear()  # free memory

        return {'val_loss': avg_loss, 'val_cmap': avg_score}



# define the logger
# wandb_logger = WandbLogger(project='Bird2023', log_model="all", name='efficientnet_b3_epoch_50')
# logger = wandb_logger


# define the data
dl_train, dl_val, ds_train, ds_val = get_fold_dls(df_train, df_valid)

# define the model
audio_model = BirdClefModel()


# early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=Config.PATIENCE, verbose= True, mode="min")
checkpoint_callback = ModelCheckpoint(dirpath=Config.save_path,
                                        save_top_k=1,
                                        save_last= True,
                                        save_weights_only=False,
                                        filename= f'./{Config.model}_loss',
                                        verbose= True,
                                        monitor='val_accuracy',
                                        mode='max',
                                        auto_insert_metric_name = True)

callbacks_to_use = [checkpoint_callback]#,early_stop_callback]

# define the trainer
trainer = Trainer(
    val_check_interval=0.5,
    deterministic=True,
    max_epochs=Config.epochs,
    # logger=logger,
    callbacks=callbacks_to_use,
    precision=Config.PRECISION, accelerator="gpu",devices=[7],
    num_sanity_val_steps=0
)

# train the model
trainer.fit(audio_model, train_dataloaders = dl_train, val_dataloaders = dl_val)


# close the wandb run and free memory
# wandb.finish()
gc.collect()
torch.cuda.empty_cache()