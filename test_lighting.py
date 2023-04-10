import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
import torchvision.transforms as transforms
import torchvision.io
import librosa as lb
from PIL import Image
import albumentations as alb
import torch.multiprocessing as mp
import warnings

warnings.filterwarnings('ignore')
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning, EarlyStopping
import torch.nn as nn
from torch.nn.functional import cross_entropy
import torchmetrics
# import timm
from pathlib import Path


class Config:
    num_classes = 264
    batch_size = 12
    PRECISION = 16
    seed = 2023
    model = "tf_efficientnet_b1_ns"
    pretrained = False
    use_mixup = False
    mixup_alpha = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = "/kaggle/input/birdclef-2023/"
    train_images = "/kaggle/input/split-creating-melspecs-stage-1/specs/train/"
    valid_images = "/kaggle/input/split-creating-melspecs-stage-1/specs/valid/"
    train_path = "/kaggle/input/bc2023-train-val-df/train.csv"
    valid_path = "/kaggle/input/bc2023-train-val-df/valid.csv"

    test_path = "test_soundscapes" # '/kaggle/input/birdclef-2023/test_soundscapes/'
    SR = 32000
    DURATION = 5
    LR = 5e-4

    model_ckpt = "ckpt/last.ckpt"#'/kaggle/input/exp2-b1-maxs7/exp1/last.ckpt'


pl.seed_everything(Config.seed, workers=True)

def config_to_dict(cfg):
    return dict((name, getattr(cfg, name)) for name in dir(cfg) if not name.startswith('__'))


def compute_melspec(y, sr, n_mels, fmin, fmax):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = lb.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )

    melspec = lb.power_to_db(melspec).astype(np.float32)
    return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])

        n_repeats = length // len(y)
        epsilon = length % len(y)

        y = np.concatenate([y] * n_repeats + [y[:epsilon]])

    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y


df_test = pd.DataFrame(
     [(path.stem, *path.stem.split("_"), path) for path in Path(Config.test_path).glob("*.ogg")],
    columns = ["filename", "name" ,"id", "path"]
)
print(df_test.shape)
df_test.head()

import albumentations as A
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
                A.Cutout(max_h_size=5, max_w_size=16),
                A.CoarseDropout(max_holes=4),
            ], p=0.5),
    ])


import librosa.display as lbd
import soundfile as sf
from soundfile import SoundFile


class BirdDataset(Dataset):
    def __init__(self, data, sr=Config.SR, n_mels=128, fmin=0, fmax=None, duration=Config.DURATION, step=None,
                 res_type="kaiser_fast", resample=True):

        self.data = data
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr // 2

        self.duration = duration
        self.audio_length = self.duration * self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def audio_to_image(self, audio):
        melspec = compute_melspec(audio, self.sr, self.n_mels, self.fmin, self.fmax)
        image = mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = []
        for i in range(self.audio_length, len(audio) + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            audios.append(audio[start:end])

        if len(audios[-1]) < self.audio_length:
            audios = audios[:-1]

        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        return images

    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "path"])


ds_test = BirdDataset(
    df_test,
    sr = Config.SR,
    duration = Config.DURATION,
)


def predict(data_loader, model):
    model.to('cpu')
    model.eval()
    predictions = []
    for en in range(len(ds_test)):
        print(en)
        images = torch.from_numpy(ds_test[en])
        print(images.shape)
        with torch.no_grad():
            outputs = model(images).sigmoid().detach().cpu().numpy()
            print(outputs.shape)
        #             pred_batch.extend(outputs.detach().cpu().numpy())
        #         pred_batch = np.vstack(pred_batch)
        predictions.append(outputs)

    return predictions


df_train = pd.read_csv(Config.train_path)
Config.num_classes = len(df_train.primary_label.unique())

print(f"Create Dataloader...")

ds_test = BirdDataset(
    df_test,
    sr = Config.SR,
    duration = Config.DURATION,
)

import sklearn.metrics

def padded_cmap(solution, submission, padding_factor=5):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score

def map_score(solution, submission):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    score = sklearn.metrics.average_precision_score(
        solution.values,
        submission.values,
        average='micro',
    )
    return score

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR

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
        model_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=Config.LR,
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

    def training_step(self, batch, batch_idx):
        image, target = batch

        y_pred = self(image)
        loss = self.loss_function(y_pred, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        y_pred = self(image)
        val_loss = self.loss_function(y_pred, target)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return {"val_loss": val_loss, "logits": y_pred, "targets": target}

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader

    def validation_epoch_end(self, outputs):
        pass
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # output_val = torch.cat([x['logits'] for x in outputs], dim=0).sigmoid().cpu().detach().numpy()
        # target_val = torch.cat([x['targets'] for x in outputs], dim=0).cpu().detach().numpy()
        #
        # # print(output_val.shape)
        # val_df = pd.DataFrame(target_val, columns=birds)
        # pred_df = pd.DataFrame(output_val, columns=birds)
        #
        # avg_score = padded_cmap(val_df, pred_df, padding_factor=5)
        # avg_score2 = padded_cmap(val_df, pred_df, padding_factor=3)
        # avg_score3 = sklearn.metrics.label_ranking_average_precision_score(target_val, output_val)
        #
        # #         competition_metrics(output_val,target_val)
        # print(f'epoch {self.current_epoch} validation loss {avg_loss}')
        # print(f'epoch {self.current_epoch} validation C-MAP score pad 5 {avg_score}')
        # print(f'epoch {self.current_epoch} validation C-MAP score pad 3 {avg_score2}')
        # print(f'epoch {self.current_epoch} validation AP score {avg_score3}')
        #
        # val_df.to_pickle('val_df.pkl')
        # pred_df.to_pickle('pred_df.pkl')
        #
        # return {'val_loss': avg_loss, 'val_cmap': avg_score}


audio_model = BirdClefModel()

print("Model Creation")

model = BirdClefModel.load_from_checkpoint(Config.model_ckpt, train_dataloader=None,validation_dataloader=None)
print("Running Inference..")

preds = predict(ds_test, model)


torch.cuda.empty_cache()


filenames = df_test.filename.values.tolist()
bird_cols = list(pd.get_dummies(df_train['primary_label']).columns)
sub_df = pd.DataFrame(columns=['row_id']+bird_cols)


for i, file in enumerate(filenames):
    pred = preds[i]
    num_rows = len(pred)
    row_ids = [f'{file}_{(i + 1) * 5}' for i in range(num_rows)]
    df = pd.DataFrame(columns=['row_id'] + bird_cols)

    df['row_id'] = row_ids
    df[bird_cols] = pred

    sub_df = pd.concat([sub_df, df]).reset_index(drop=True)


sub_df.to_csv('submission.csv',index=False)