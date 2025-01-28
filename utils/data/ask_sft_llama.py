import pandas as pd
import numpy as np
import pickle

with open('convert_dict.pkl', 'rb') as f:
    overall_dict = pickle.load(f)
    
item_content = pd.read_csv('raw-data.csv')
cold_item_content = pd.DataFrame(columns=['item_id', 'content'])
for i in range(len(overall_dict['cold_item'])):
    cold_item_content.loc[i] = [overall_dict['cold_item'][i], item_content.iloc[overall_dict['cold_item'][i]]['title']]

cold_user_list = pd.read_csv("C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\\final_data\CiteULike\CiteULike_Morec_top20.csv")
cold_user_list = cold_user_list[['item_id', 'user_id']]
cold_user_list_dict = cold_user_list.groupby('item_id')['user_id'].apply(list)

with open('train_user_preference_list.pkl', 'rb') as f:
    train_user_preference_list = pickle.load(f)

from tqdm import tqdm
citeUlike_cold_prompt_list = []
for key, value in tqdm(cold_user_list_dict.items()):
    cold_item_content_id = key
    for j in value:
        cold_prompt = {}
        user_id = j
        prompt = \
            f'''Given the user's interaction paper set, determine whether the user will like the target paper by answering \"Yes\" or \"No.\"\nUser preference:\"{train_user_preference_list[user_id]}\"\n,Whether the user will like the target paper \"{item_content.iloc[key]['title']}\"?'''
        cold_prompt['user_id'] = user_id
        cold_prompt['cold_item_content_id'] = cold_item_content_id
        cold_prompt['prompt'] = prompt
        citeUlike_cold_prompt_list.append(cold_prompt)

with open('your_pkl_file_name', 'wb') as f:
    pickle.dump(citeUlike_cold_prompt_list, f)  
