def get_topn(topn,ALDI_item_embedding_file_path,ALDI_user_embedding_file_path,llama_tower_item_embedding_file_path,llama_tower_user_embedding_file_path,convert_dict,save_file_name):
    import numpy as np
    import torch
    import pickle
    ALDI_item_embedding = torch.from_numpy(np.load(ALDI_item_embedding_file_path))
    ALDI_user_embedding = torch.from_numpy(np.load(ALDI_user_embedding_file_path))
    llama_tower_item_embedding = torch.load(llama_tower_item_embedding_file_path)
    llama_tower_user_embedding = torch.load(llama_tower_user_embedding_file_path)
    with open(convert_dict, 'rb') as f:
        para_dict = pickle.load(f)


    concat_item_emb = torch.cat((ALDI_item_embedding, llama_tower_item_embedding), dim=1)

    concat_user_emb = torch.cat((ALDI_user_embedding[1], llama_tower_user_embedding), dim=1)
    from tqdm import tqdm
    cold_item_set_dict = {}
    user_tensor = concat_user_emb
    print('开始计算')
    for i in tqdm(para_dict['cold_item']):
        item_tensor = concat_item_emb[i]
        result = item_tensor @ user_tensor.t()
        top_k_values, top_k_indices = torch.topk(result, k=topn, largest=True)
        cold_item_set_dict[i] = top_k_indices

    triplets = []
    print('开始生成文件')
    for key, value in tqdm(cold_item_set_dict.items()):
        for user in value:
            triple = []
            triple.append(key)
            triple.append(int(user))
            triple.append(1)
            triplets.append(triple)
    import csv
    csv_file_name = save_file_name
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['item_id', 'user_id', 'clk/no'])
        for triplet in triplets:
            writer.writerow(triplet)

def calculate_caina(before_ask_file_path, after_ask_file_path):
    import pandas as pd
    before_ask = pd.read_csv(before_ask_file_path)
    print(len(before_ask))
    after_ask = pd.read_csv(after_ask_file_path)
    print(len(after_ask))
    return (len(after_ask) - 1)/(len(before_ask) - 1)


if __name__ == '__main__':
    before_ask_file_path = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\\final_data\\ab_0116\\top20\CiteULike_wide20.csv'
    after_ask_file_path = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\\final_data\\ab_0116\\askllama\gpt3.5.csv'
    random_20 = calculate_caina(before_ask_file_path, after_ask_file_path)
    print(random_20)