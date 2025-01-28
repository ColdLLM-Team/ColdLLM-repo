import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
checkpoint = '../../export/llama2_AmazonBaby_0407_v1/'


class LlamaTower(nn.Module):
    def __init__(self, ):
        super(LlamaTower, self).__init__()
        self.llama_encoder = AutoModel.from_pretrained(checkpoint)
        self.user_mlp = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 200),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 200),
        )

    def forward(self, user_text, item_text):
        user_emb = self.llama_encoder(**user_text)[0]
        item_emb = self.llama_encoder(**item_text)[0]
        user_content_emb = torch.mean(user_emb, dim=1)
        item_content_emb = torch.mean(item_emb, dim=1)

        return user_content_emb, item_content_emb


import ijson
from torch.utils.data import Dataset, random_split
import json


class CiteULike(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            objects = ijson.items(f, 'item')
            for idx, obj in enumerate(objects):
                Data[idx] = obj
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


user_data = CiteULike(
    '/raid/home/zhenghangyang/LLaMA-Factory/export/Llama_Tower/Amazon/baby/baby4LlamaTower_all_user.json')
item_data = CiteULike(
    '/raid/home/zhenghangyang/LLaMA-Factory/export/Llama_Tower/Amazon/baby/baby4LlamaTower_all_item.json')
print(next(iter(user_data)))
print(next(iter(item_data)))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token


def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['user_info'])
        batch_sentence_2.append(sample['item_info'])
        batch_label.append(int(sample['label']))

    X_1 = tokenizer(
        batch_sentence_1,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    X_2 = tokenizer(
        batch_sentence_2,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    y = torch.tensor(batch_label)
    return X_1, X_2, y


user_dataloader = DataLoader(user_data, batch_size=16, collate_fn=collote_fn)
item_dataloader = DataLoader(item_data, batch_size=16, collate_fn=collote_fn)

batch_X1, batch_X2, batch_y = next(iter(user_dataloader))
print('batch_X1 shape:', {k: v.shape for k, v in batch_X1.items()})
print('batch_X2 shape:', {k: v.shape for k, v in batch_X2.items()})
print('batch_y shape:', batch_y.shape)
print(batch_X1)
print(batch_X2)
print(batch_y)

model = LlamaTower()
device = torch.device('cuda:2')
model = model.to(device)
model_device = next(model.parameters()).device
print(f'model_device is {model_device}')

from tqdm import tqdm
model.eval()
user_content_emb_list = []
item_content_emb_list = []
with torch.no_grad():
    for X1, X2, y in tqdm(user_dataloader):
        model_device = next(model.parameters()).device
        X1, X2, y = X1.to(device),X2.to(device),y.to(device)
        user_content_emb, _ = model(X1,X2) 
        user_content_emb_list.append(user_content_emb.cpu())

user_content_emb_tensor = torch.cat(user_content_emb_list, dim=0)

torch.save(user_content_emb_tensor, 'user_content_emb_tensor_baby.pt')
print(f'user_content_emb_tensor shape is {user_content_emb_tensor.shape}')
print('user over item start')
with torch.no_grad():
    for X1, X2, y in tqdm(item_dataloader):
        X1, X2, y = X1.to(device),X2.to(device),y.to(device)
        _,item_content_emb = model(X1,X2)  
        item_content_emb_list.append(item_content_emb.cpu())
item_content_emb_tensor = torch.cat(item_content_emb_list, dim=0)
torch.save(item_content_emb_tensor, 'item_content_emb_tensor_baby.pt')
print(f'item_content_emb_tensor shape is {item_content_emb_tensor.shape}')
print('item over')