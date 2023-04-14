from torch.utils.data import Dataset, DataLoader
import pickle, numpy as np
import torch
import random
from utils import load_pickle
from torchvision.utils import tshow
import librosa
import soundfile as sf


class MyDataset(Dataset):
    def __init__(self, data, mode):
        assert mode in ['train', 'test', 'val'], f'invalid mode {mode}!, mode must be [train | val | test]'
        self.mode = mode
        self.data = data
        # self.sr = 32000  # test use only
        # self.duration = 5 # test use only
        # self.audio_length = self.duration * self.sr # test use only

        self.val_fold_idx = 0
        k_fold = 5
        data_total_len = len(data['primary_label'])
        all_indices = list(range(data_total_len))
        random.seed(42)
        random.shuffle(all_indices)

        self.val_indices = all_indices[self.val_fold_idx*data_total_len//k_fold: (self.val_fold_idx+1)*data_total_len//k_fold]
        self.train_indices = [idx for idx in all_indices if idx not in self.val_indices]

        train_labels = [data['primary_label'][i] for i in self.train_indices]
        val_labels = [data['primary_label'][i] for i in self.val_indices]

        missing_labels = list(set(val_labels)-set(train_labels)) # labels in val set but not in train set
        val_indices_copy = self.val_indices.copy()
        for i, val_idx in enumerate(val_indices_copy):
            if data['primary_label'][val_idx] in missing_labels:
                self.train_indices.append(self.val_indices.pop(i))


    def __getitem__(self, index):
        if self.mode == 'train':
            mel = self.data['mel'][self.train_indices[index]]
            label = torch.tensor(self.data['primary_label_idx'][self.train_indices[index]])

        elif self.mode == 'val':
            mel = self.data['mel'][self.val_indices[index]]
            label = torch.tensor(self.data['primary_label_idx'][self.val_indices[index]])
        else:
            raise

        mel = self.crop_or_pad(mel)
        label = torch.nn.functional.one_hot(label, num_classes=264).float()

        return mel, label

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_indices)
        elif self.mode == 'val':
            return len(self.val_indices)

    def crop_or_pad(self, m, th=313):  # 313=5s*32000Hz/512
        length = m.shape[1]
        if length <= th: # pad short
            while m.shape[1] < th:  # repeat padding until th
                m = np.concatenate([m, m],axis=1)
            m = m[:,0:th]
        else:  # crop longer audio
            start = np.random.randint(length - th)
            m = m[:,start:start+th]
        return torch.from_numpy(m).unsqueeze(0)


