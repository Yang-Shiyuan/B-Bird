
import torch, torch.nn as nn
from model import MyModel
from dataset import MyDataset
from utils import load_pickle, padded_cmap
import pandas as pd  #
from pathlib import Path

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'  # use gpu if available
ckpt_resume = "ckpt/mv2(0.65).ckpt"
test_data_path = "test_soundscapes"


df_test = pd.DataFrame(
     [(path.stem, *path.stem.split("_"), path) for path in Path(test_data_path).glob("*.ogg")],
    columns = ["filename", "name" ,"id", "path"]
)
print(df_test.shape)
df_test.head()

test_dataset = MyDataset(data=df_test, mode='test')
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

my_model = MyModel(num_classes=264).to(device)
my_model.load_state_dict(torch.load(ckpt_resume, map_location='cpu'), strict=False)
print(f"ckpt resumed from {ckpt_resume}")
my_model.eval()

predictions = []
with torch.no_grad():
    for test_idx in range(len(test_dataset)):
        test_sample_np = test_dataset[test_idx]  # test_sample_np is in size of (120,128,313)
        test_sample_pt = torch.from_numpy(test_sample_np).unsqueeze(1).to(device)  # test_sample_pt is in size of (120,1,128,313)
        pred = my_model(test_sample_pt).sigmoid().detach().cpu().numpy()
        predictions.append(pred)

bird_names = load_pickle("bird_names.pickle3")  # bird_names is a list containing 264 bird names
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