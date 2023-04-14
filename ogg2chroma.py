import pandas as pd
from collections import OrderedDict
import librosa,pickle
import  numpy as np
from tqdm import tqdm

# read train_metadata.csv
csv_data = pd.read_csv('train_metadata.csv')
sr = 32000

# convert csv to dict
data_dict=OrderedDict()
for col in csv_data.columns:
    data_dict[col]=csv_data[col].tolist()

# add key "mel" for storing mel
data_dict['chroma'] = [0] * len(data_dict['primary_label'])

# convert audio to mel (THIS COSTS ~1 HOUR)
for idx, ogg_path in tqdm(enumerate(data_dict['filename'])):
    x, sr = librosa.load(f"train_audio/{ogg_path}", sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr, n_chroma=12, n_fft=4096)
    data_dict['chroma'][idx] = chroma_stft

# assign number_id to each bird, stored in a dict name_label_2_int_label, i.e., name_label_2_int_label['bird_name'] = id
bird_names, _ = np.unique(data_dict['primary_label'], return_counts=True)
name_label_2_int_label = {}
for idx, k in enumerate(bird_names):
    name_label_2_int_label[k] = idx

# add key "primary_label_idx" for saving primary_label in num_id format
primary_label_idx = [0] * len(data_dict['primary_label'])
for i, tmp in enumerate(data_dict['primary_label']):
    primary_label_idx[i] = name_label_2_int_label[tmp]
data_dict['primary_label_idx'] = primary_label_idx
# add key "secondary_label_idx" for saving primary_label in num_id format
import ast

secondary_label_idx = [0] * len(data_dict['primary_label'])
for i, tmp in enumerate(data_dict['secondary_labels']):
    if tmp == '[]':
        secondary_label_idx[i] = []
    else:
        tmp_label_list = ast.literal_eval(tmp)
        secondary_label_idx[i] = [name_label_2_int_label[p] for p in tmp_label_list]
data_dict['secondary_label_idx'] = secondary_label_idx

# save our data dict into pickle file (~22GB)
with open("train_chroma(k=12).pickle3", 'wb') as file:
    pickle.dump(data_dict, file)