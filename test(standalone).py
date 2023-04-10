import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import pickle, numpy as np
import torch
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path

## for local
device = 'cpu'
ckpt_resume = "ckpt/mv2(0.65).ckpt"
test_data_path = "test_soundscapes"
name_label_2_int_label_pickle_path = "name_label_2_int_label.pickle3"
bird_names_pickle_path = "bird_names.pickle3"

## for kaggle submission
# device = 'cpu'
# ckpt_resume = "/kaggle/input/stuffs/mv2(0.65).ckpt"
# test_data_path = "/kaggle/input/birdclef-2023/test_soundscapes"
# name_label_2_int_label_pickle_path = "/kaggle/input/stuffs/name_label_2_int_label.pickle3"
# bird_names_pickle_path = "/kaggle/input/stuffs/bird_names.pickle3"


def load_pickle(fname):
    f = open(fname, 'rb')
    out = pickle.load(f)
    f.close()
    return out


class MyTestDataset(Dataset):
    def __init__(self, data):
        self.sr = 32000
        self.duration = 5
        self.audio_length = self.duration * self.sr
        self.name_label_2_int_label = load_pickle(name_label_2_int_label_pickle_path)  # a dict which saves mapping from
        self.data_test_df = data

    def __getitem__(self, index):
        return self.read_file(self.data_test_df.loc[index, "path"])

    def __len__(self):
        return len(self.data_test_df)

    def audio_to_image(self, audio, ):
        melspec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128, fmin=0, fmax=self.sr//2)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        # image = mono_to_color(melspec)
        # image = self.normalize(image)
        return melspec

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")
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


class MyModel(nn.Module):
    def __init__(self, num_classes=264):
        super(MyModel, self).__init__()
        self.mv2 = torchvision.models.mobilenet_v2()
        self.classifier =nn.Sequential(
            nn.Dropout(0.4),nn.Linear(1000, 1000),nn.ReLU(),
            nn.Dropout(0.4),nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.mv2(x.repeat(1,3,1,1))  # [b, 3, f=12|128, t=4096]
        x = self.classifier(x)
        return x


df_test = pd.DataFrame(
     [(path.stem, *path.stem.split("_"), path) for path in Path(test_data_path).glob("*.ogg")],
    columns = ["filename", "name" ,"id", "path"]
)
df_test.head()

test_dataset = MyTestDataset(data=df_test)

my_model = MyModel(num_classes=264).to(device)
my_model.load_state_dict(torch.load(ckpt_resume, map_location='cpu'), strict=False)
print(f"ckpt resumed from {ckpt_resume}")
my_model.eval()

predictions = []
with torch.no_grad():
    for test_idx in range(len(test_dataset)):
        test_sample_np = test_dataset[test_idx]  # test_sample_np is in size of (120,128,313)
        test_sample_pt = torch.from_numpy(test_sample_np).unsqueeze(1).to(device)  # test_sample_pt is in size of (120,1,128,313)
        pred = my_model(test_sample_pt).softmax().detach().cpu().numpy()
        predictions.append(pred)

bird_names = load_pickle(bird_names_pickle_path)  # bird_names is a list containing 264 bird names
submission_df = pd.DataFrame(columns=['row_id']+bird_names)  # csv head
test_filenames = df_test.filename.values.tolist()

for test_idx, test_file in enumerate(test_filenames):
    pred = predictions[test_idx]
    num_rows = len(pred)
    row_ids = [f'{test_file}_{(i + 1) * 5}' for i in range(num_rows)]
    current_df = pd.DataFrame(columns=['row_id'] + bird_names)
    current_df['row_id'] = row_ids
    current_df[bird_names] = pred
    submission_df = pd.concat([submission_df, current_df]).reset_index(drop=True)

submission_df.to_csv('submission.csv',index=False)
print("done")