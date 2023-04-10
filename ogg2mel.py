import pandas as pd
from collections import OrderedDict
from torchvision.utils import tshow
import librosa, os, pickle
import torch, numpy as np, matplotlib.pyplot as plt  # 建议matplotlib版本3.3.4
from PIL import Image
from tqdm import tqdm
csv_data = pd.read_csv('train_metadata.csv')
sr = 32000


train_dict=OrderedDict()
for col in csv_data.columns:
    train_dict[col]=csv_data[col].tolist()

train_dict['mel'] = [0]*len(train_dict['primary_label'])

for idx, ogg_path in tqdm(enumerate(train_dict['filename'])):
    x, sr = librosa.load(f"train_audio/{ogg_path}", sr=sr)
    m = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128, fmin=0, fmax=sr//2)
    ## either
    # m = np.log(m + 1e-20)
    # train_dict['mel'][idx] = m
    ## or
    m = librosa.power_to_db(m).astype(np.float32)
    train_dict['mel'][idx] = m


with open("train_mel(dB,sr=32k,bin=128).pickle3", 'wb') as file:
    pickle.dump(train_dict, file)

# x, sr = librosa.load(mp3, sr=8000)
# x_ = librosa.effects.pitch_shift(y=x, sr=sr, n_steps=2)
# m_ = librosa.feature.melspectrogram(y=x_, sr=sr, n_mels=128)
# mels_[idx] = m_
# if idx <= 5:
#     sf.write(f"{mp3_info['fname']}.wav", x, sr, 'PCM_24')
#     sf.write(f"{mp3_info['fname']}-.wav", x_, sr, 'PCM_24')
#
# with open("musicdata/train_chromas.pickle3", 'wb') as file:
#     pickle.dump(mels_, file)