import torch
llama_tower_user_embedding = torch.load('your_path')
llama_tower_item_embedding = torch.load('your_path')

import torch
morec_user_embedding = torch.load('your_path', map_location=torch.device('cpu'))
morec_item_embedding = torch.load('your_path', map_location=torch.device('cpu'))

morec_user_embedding = morec_user_embedding[:-1]
morec_item_embedding = morec_item_embedding[:-1]

import pickle
with open('convert_dict.pkl', 'rb') as f:
    para_dict = pickle.load(f)
cold_item_embedding = llama_tower_item_embedding[para_dict['cold_item']]

cold_item_set_dict = {}
user_tensor = morec_user_embedding
for i in para_dict['cold_item']:
    item_tensor = morec_item_embedding[i]
    result = item_tensor @ user_tensor.t()
    top_k_values, top_k_indices = torch.topk(result, k=20, largest=True)
    cold_item_set_dict[i] = top_k_indices
result = llama_tower_item_embedding[para_dict['cold_item']] @ llama_tower_user_embedding.t()

top_k_values, top_k_indices = torch.topk(result, k=20, largest=True)

triplets = []
for key,value in cold_item_set_dict.items():
    for user in value:
        triple = []
        triple.append(key)
        triple.append(int(user))
        triple.append(1)
        triplets.append(triple)

import csv
csv_file_name = 'CiteULike_top20.csv'
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=',')

    writer.writerow(['item_id', 'user_id', 'clk/no'])

    for triplet in triplets:
        writer.writerow(triplet)

