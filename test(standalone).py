import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import pickle, numpy as np
import torch
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import timm
## for local
device = 'cpu'
ckpt = "ckpt/tf_efficientnet_b2_ns_loss.ckpt"
test_data_path = "test_soundscapes"
name_label_2_int_label_pickle_path = "name_label_2_int_label.pickle3"
bird_names_pickle_path = "bird_names.pickle3"

## for kaggle submission
# device = 'cpu'
# ckpt = "/kaggle/input/stuffs/mv2(0.65).ckpt"
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
        self.enable_resample = True  # enable resampling to sr if the actual sr is not sr
        self.duration = 5  # 5s
        self.audio_length = self.duration * self.sr
        self.name_label_2_int_label = load_pickle(name_label_2_int_label_pickle_path)  # a dict which saves mapping from
        self.data_test_df = data

    def __getitem__(self, index):
        return self.read_file(self.data_test_df.loc[index, "path"])

    def __len__(self):
        return len(self.data_test_df)

    def audio_to_image(self, audio):
        melspec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128, fmin=0, fmax=self.sr//2)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        melspec = self.normalize(melspec)
        melspec = torch.from_numpy(melspec).float().unsqueeze(0).repeat(3, 1, 1)
        return melspec

    def normalize(self, X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean()
        std = std or X.std()
        X = (X - mean) / (std + eps)
        _min, _max = X.min(), X.max()
        if (_max - _min) > eps:
            V = np.clip(X, _min, _max)
            V = (V - _min) / (_max - _min)
        else:
            V = np.zeros_like(X)
        return V

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")
        if self.enable_resample and orig_sr != self.sr:
            audio = librosa.resample(audio, orig_sr, self.sr, res_type="kaiser_fast")

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

#################################### Define Model here ###########################################
# class MyModel(nn.Module):
#     def __init__(self, num_classes=264):
#         super(MyModel, self).__init__()
#         self.mv2 = torchvision.models.mobilenet_v2()
#         self.classifier =nn.Sequential(
#             nn.Dropout(0.4),nn.Linear(1000, 1000),nn.ReLU(),
#             nn.Dropout(0.4),nn.Linear(1000, num_classes))
#
#     def forward(self, x):
#         x = self.mv2(x)  # [b, 3, f=12|128, t=4096]
#         x = self.classifier(x)
#         return x


class MyModel(nn.Module):
    def __init__(self,  num_classes=264):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model("tf_efficientnet_b2_ns", pretrained=False)
        self.in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.in_features, num_classes)
        )

    def forward(self, images):
        logits = self.backbone(images)
        return logits

my_model = MyModel(num_classes=264).to(device)
my_model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
print(f"ckpt resumed from {ckpt}")
my_model.eval()

############################   Read Test Set       ##################################
df_test = pd.DataFrame(
     [(path.stem, *path.stem.split("_"), path) for path in Path(test_data_path).glob("*.ogg")],
    columns = ["filename", "name" ,"id", "path"]
)
df_test.head()
test_dataset = MyTestDataset(data=df_test)


############################  Make Inference     ##################################

predictions = []
with torch.no_grad():
    for test_idx in range(len(test_dataset)):
        test_sample_np = test_dataset[test_idx]  # test_sample_np is an ogg's sliced mel images with the size of (120,128,313)
        test_sample_pt = torch.from_numpy(test_sample_np).to(device)  # numpy to torch
        pred = my_model(test_sample_pt).sigmoid().detach().cpu().numpy()
        predictions.append(pred)


############################  Write Results   ##################################

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

''' sanity check
import sklearn.metrics
pseudo_label = np.load("/home/yangshiyuan/Projects/birdclef-2023/test_soundscapes/pseudo_label.npy")
avg_score = sklearn.metrics.label_ranking_average_precision_score(pseudo_label, predictions[0])
print(avg_score)

pad=5
cmap=sklearn.metrics.average_precision_score(
    np.concatenate([pseudo_label,np.ones((pad,264))]),
    np.concatenate([predictions[0],np.ones((pad,264))]),
    average='macro',
)
print(cmap)
'''
