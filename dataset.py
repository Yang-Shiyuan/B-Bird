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

        self.sr = 32000  # test use only
        self.duration = 5 # test use only
        self.audio_length = self.duration * self.sr # test use only

        # self.name_label_2_int_label = load_pickle("name_label_2_int_label.pickle3")  # a dict which saves mapping from
        # bird name to label index, can be generated using the code below:
        # bird_names, _ = np.unique(data['primary_label'], return_counts=True)
        # self.name_label_2_int_label = {}
        # for idx, k in enumerate(bird_names):
        #     self.name_label_2_int_label[k] = torch.tensor(idx,dtype=torch.long)

        if mode == 'train':
            total_data_length = len(data['primary_label'])
            self.data = {k: v[:int(0.8*total_data_length)] for k, v in data.items()}  # 0~80% as train set
        elif mode == 'val':
            total_data_length = len(data['primary_label'])
            self.data = {k: v[int(0.8 * total_data_length):] for k, v in data.items()}  # 80%~100% as val set
        elif mode == 'test':  # in test mode, data is in DataFrame form
            self.data_test_df = data
        else:
            raise ValueError(f'no such mode {mode}')

    def __getitem__(self, index):
        if self.mode == 'train' or  self.mode == 'val':
            mel = self.data['mel'][index]
            mel = self.crop_or_pad(mel)
            # label = self.name_label_2_int_label[self.data['primary_label'][index]]  # 'bird name' -> idx
            label = torch.tensor(self.data['primary_label_idx'][index])
            label = torch.nn.functional.one_hot(label, num_classes=264).float()
            return mel, label

        elif self.mode == 'test':
            return self.read_file(self.data_test_df.loc[index, "path"])


    def __len__(self):
        if self.mode == 'train' or  self.mode == 'val':
            return len(self.data['primary_label'])
        elif self.mode == 'test':
            return len(self.data_test_df)

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

    ########## following methods are for test use only ##################

    def audio_to_image(self, audio, ):
        melspec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128, fmin=0, fmax=self.sr//2)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        # image = mono_to_color(melspec)
        # image = self.normalize(image)
        return melspec

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        # if self.resample and orig_sr != self.sr:
        #     audio = librosa.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = []
        for i in range(self.audio_length, len(audio) + self.audio_length, self.audio_length):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            audios.append(audio[start:end])

        if len(audios[-1]) < self.audio_length:
            audios = audios[:-1]

        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)
        return images


