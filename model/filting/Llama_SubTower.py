# Llama-SubTower
import torch
from torch import nn
from tqdm import tqdm

class Llama_head(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Llama_head, self).__init__()
        self.user_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, user_origin_emb, item_origin_emb):
        user_content_emb = self.user_mlp(user_origin_emb)
        item_content_emb = self.item_mlp(item_origin_emb)
        sim = torch.matmul(user_content_emb,item_content_emb.t())
        logits = torch.sigmoid(torch.diag(sim))
        return logits

interaction_file_path = 'warm_emb.csv'
train_interaction_file_path = 'warm_emb.csv'
import pandas as pd
interaction_df = pd.read_csv(interaction_file_path)
train_interaction_df = pd.read_csv(train_interaction_file_path)
item_origin_emb = torch.load('item_content_emb_tensor_200_ml-1m.pt')

user_origin_emb = torch.load('user_content_emb_tensor_200_ml-1m.pt')

import  numpy as np
group_df = train_interaction_df.groupby('user')
all_interaction = []
for name, group in tqdm(group_df):

    for value in group['item'].values:
        one_user_pos_interaction = []
        one_user_pos_interaction.append(user_origin_emb[name])
        one_user_pos_interaction.append(item_origin_emb[value])
        one_user_pos_interaction.append(1)
        all_interaction.append(one_user_pos_interaction)

    set_all = set(range(0, len(item_origin_emb)))
    set_neg = set_all - set(group['item'].values)
    if len(set_neg) >= len(group['item'].values):
        neg_samples = np.random.choice(list(set_neg), size=len(group['item'].values))
        # print(neg_samples)
        for neg_value in neg_samples:
            one_user_neg_interaction = []
            one_user_neg_interaction.append(user_origin_emb[name])
            one_user_neg_interaction.append(item_origin_emb[neg_value])
            one_user_neg_interaction.append(0)
            all_interaction.append(one_user_neg_interaction)
    elif len(set_neg) > 0:
        neg_samples = np.random.choice(list(set_neg), size=len(set_neg))
        for neg_value in neg_samples:
            one_user_neg_interaction = []
            one_user_neg_interaction.append(user_origin_emb[name])
            one_user_neg_interaction.append(item_origin_emb[neg_value])
            one_user_neg_interaction.append(0)
            all_interaction.append(one_user_neg_interaction)
    else:
        continue

interaction_df_from_list = pd.DataFrame(all_interaction, columns=['user_emb','item_emb','label'])

import torch
from torch.utils.data import Dataset, DataLoader, random_split

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        user_emb = torch.tensor(row['user_emb']).float()
        item_emb = torch.tensor(row['item_emb']).float()
        label = torch.tensor(row['label']).float()
        return user_emb, item_emb, label


dataset = CustomDataset(interaction_df_from_list)
total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

from tqdm import tqdm
def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch - 1) * len(dataloader)

    model.train()
    for step, (X1, X2, y) in enumerate(dataloader, start=1):
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        pred = model(X1, X2).float()
        y = y.float()
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    # size = len(dataloader.dataset)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X1, X2, y in dataloader:
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            pred = model(X1, X2).float()
            pred = torch.where(pred > 0.5, torch.tensor(1.0, device=pred.device), torch.tensor(0.0, device=pred.device))
            total += y.size(0)
            # print(f'total is {total}')
            # print(f'y is {y}')
            # print(f'pred is {pred}')
            correct += (pred == y).type(torch.float).sum().item()
            # print(f'correct is {correct}')
    acc = correct / total
    print(f"{mode} Accuracy: {(100 * acc):>0.1f}%\n")
    return acc


device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
model = Llama_head(4096, 2048, 200)
from transformers import AdamW, get_schedulerwei

learning_rate = 1e-5
epoch_num = 30

loss_fn = nn.BCELoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)
best_acc = 0.
total_loss = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_acc = test_loop(test_dataloader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'ml_1m_epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
print("Done!")
