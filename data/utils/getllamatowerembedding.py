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
        # sim = torch.matmul(user_content_emb,item_content_emb.t())
        # logits = torch.sigmoid(torch.diag(sim))
        return user_content_emb, item_content_emb
# interaction_file_path = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\ml-1m\\all.csv'
# import pandas as pd
# interaction_df = pd.read_csv(interaction_file_path)
item_origin_emb = torch.load('C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\item_content_emb_tensor_baby_4096.pt')

user_origin_emb = torch.load('C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\\user_content_emb_tensor_baby_4096.pt')


from torch.utils.data import Dataset, DataLoader, random_split
user_fake_emb = torch.rand(item_origin_emb.shape)
item_fake_emb = torch.rand(user_origin_emb.shape)


import pandas as pd
user_fake_all_interaction = []
for i in range(len(user_origin_emb)):
    one_user_pos_interaction = []
    one_user_pos_interaction.append(user_origin_emb[i])
    one_user_pos_interaction.append(item_fake_emb[i])
    one_user_pos_interaction.append(1)
    user_fake_all_interaction.append(one_user_pos_interaction)
user_interaction_df_from_list = pd.DataFrame(user_fake_all_interaction, columns=['user_emb','item_emb','label'])
item_fake_all_interaction = []
for i in range(len(item_origin_emb)):
    one_user_pos_interaction = []
    one_user_pos_interaction.append(user_fake_emb[i])
    one_user_pos_interaction.append(item_origin_emb[i])
    one_user_pos_interaction.append(1)
    item_fake_all_interaction.append(one_user_pos_interaction)
item_interaction_df_from_list = pd.DataFrame(item_fake_all_interaction, columns=['user_emb','item_emb','label'])


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



userdataset = CustomDataset(user_interaction_df_from_list)
itemdataset = CustomDataset(item_interaction_df_from_list)

model = Llama_head(4096, 2048, 200)
model.load_state_dict(torch.load('C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\\baby_epoch_28_valid_acc_83.8_model_weights.bin',map_location=torch.device('cpu')))
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
userdataloader = DataLoader(userdataset, batch_size=128, shuffle=False)
itemdataloader = DataLoader(itemdataset, batch_size=128, shuffle=False)
from tqdm import tqdm
model.eval()
user_content_emb_list = []
item_content_emb_list = []
with torch.no_grad():
    for X1, X2, y in tqdm(userdataloader):
        model_device = next(model.parameters()).device
        X1, X2, y = X1.to(device),X2.to(device),y.to(device)
        user_content_emb, _ = model(X1,X2)  
        user_content_emb_list.append(user_content_emb.cpu())

user_content_emb_tensor = torch.cat(user_content_emb_list, dim=0)
torch.save(user_content_emb_tensor, 'user_content_emb_tensor_200_AmazonBaby.pt')
print(f'user_content_emb_tensor shape is {user_content_emb_tensor.shape}')
print('user over item start')
with torch.no_grad():
    for X1, X2, y in tqdm(itemdataloader):
        X1, X2, y = X1.to(device),X2.to(device),y.to(device)
        _,item_content_emb = model(X1,X2)  
        item_content_emb_list.append(item_content_emb.cpu())
item_content_emb_tensor = torch.cat(item_content_emb_list, dim=0)
torch.save(item_content_emb_tensor, 'item_content_emb_tensor_200_AmazonBaby.pt')
print(f'item_content_emb_tensor shape is {item_content_emb_tensor.shape}')
print('item over')
