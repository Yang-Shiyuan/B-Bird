
# from torchvision.utils import tshow
import matplotlib.pyplot as plt
import matplotlib
import  numpy  as np
import sklearn
from scipy import stats
import csv
from scipy import io
import pickle
import os
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm
from model import MyModel
from dataset2 import MyDataset
from utils import load_pickle, padded_cmap, map_score
import pandas as pd  #


batch_size = 8
lr = 1e-5
num_epoch = 25
device = 'cuda:5' if torch.cuda.is_available() else 'cpu'  # use gpu if available
ckpt_resume = ""  # "ckpt/mv2(0.65).ckpt"
ckpt_out = "ckpt"
os.makedirs(ckpt_out, exist_ok=True)

try:
    # data = load_pickle("train_mel(dB,sr=32k,bin=128).pickle3")  # see ogg2mel for how to convert ogg to mel_dict.pickle3
    data = load_pickle("/home/yangshiyuan/Projects/birdclef-2023/train_mel(dB,sr=32k,bin=128).shuffled_subset1.pickle3")  # see ogg2mel for how to convert ogg to mel_dict.pickle3
    bird_names, _ = np.unique(data['primary_label'], return_counts=True)
except FileNotFoundError:
    print("dataset not found! you can generate one by using ogg2mel.py")


train_dataset = MyDataset(data=data, mode='train')
val_dataset = MyDataset(data=data, mode='val')

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

my_model = MyModel(num_classes=264).to(device)

try:
    my_model.load_state_dict(torch.load(ckpt_resume, map_location='cpu'), strict=False)
    print(f"ckpt resumed from {ckpt_resume}")
except FileNotFoundError:
    print(f"ckpt not founded! Re-train from original model")


optimizer = torch.optim.Adam(my_model.parameters(), lr=lr)
loss_func = torch.nn.BCEWithLogitsLoss()

for ep in range(0, num_epoch):
    num_correct = 0
    acc=0
    tqdm_iterator = tqdm.tqdm(train_dataloader)
    for i, (x, y) in enumerate(tqdm_iterator):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = my_model(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()

        correct_current_iter = (torch.argmax(y_pred,dim=1)==torch.argmax(y,dim=1)).sum().item()
        num_correct += correct_current_iter
        acc = num_correct/(train_dataloader.batch_size*(i+1))

        tqdm_iterator.set_description(f'ep={ep}, iter={i}, loss={loss.item():.4f}, \
        correct in current iter=: {correct_current_iter}/{train_dataloader.batch_size}, acc (so far)={acc:.2f}')

    print(f'training completed at ep={ep:02}')
    #
    with torch.no_grad(): # evaluation
        print(f'eval ....')
        num_correct = 0
        acc = 0
        my_model.eval()
        y_perds = []
        y_labels = []
        loss_val = []
        for i, (x, y) in enumerate(val_dataloader):
            x = x.to(device)
            y = y.to(device)
            y_pred = my_model(x)
            loss_val.append(loss_func(y_pred, y).detach().cpu().numpy())

            y_perds.append(y_pred.sigmoid().detach().cpu().numpy())
            y_labels.append(y.cpu().numpy())

            correct_current_iter = (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item()
            num_correct += correct_current_iter
            acc = num_correct / (val_dataloader.batch_size * (i + 1))
            # print(f'val_ep={ep}, iter={i}, correct in current iter=: {correct_current_iter}/{val_dataloader.batch_size}, acc (so far)={acc:.2f}')

        y_perds_np = np.concatenate(y_perds)
        y_labels_np = np.concatenate(y_labels)
        loss_val_mean = np.array(loss_val).mean()

        # val_pred_df = pd.DataFrame(y_perds_np, columns=bird_names)
        # val_label_df = pd.DataFrame(y_labels_np, columns=bird_names)
        #score = padded_cmap(val_label_df, val_pred_df, padding_factor= 5)
        # score_map = map_score(val_label_df, val_pred_df)

        avg_score = sklearn.metrics.label_ranking_average_precision_score(y_labels_np, y_perds_np)


        print(f'eval completed at ep={ep:02}, ap={avg_score:.4f}, avg_loss={loss_val_mean:.4f} -----------------')

        my_model.train()