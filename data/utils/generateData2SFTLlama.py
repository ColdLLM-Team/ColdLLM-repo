import os
from tqdm import tqdm
import pandas as pd
def saveUserPreference(item_num, datasetdir_path, item_raw_data_filename, instruction) -> None:
    train_file_name = 'warm_emb.csv'
    train_file_path = os.path.join(datasetdir_path, train_file_name)
    train_file = pd.read_csv(train_file_path)
    train_file_groupby_user = train_file.groupby('user')
    train_file_groupby_user_list = [list(train_file_groupby_user.get_group(x).item) for x in
                                    train_file_groupby_user.groups]

    item_raw_data_path = os.path.join(datasetdir_path, item_raw_data_filename)
    item_raw_data = pd.read_csv(item_raw_data_path)

    item_raw_data = item_raw_data.fillna('')
    all_interaction_file = os.path.join(datasetdir_path, 'all.csv')
    all_interaction = pd.read_csv(all_interaction_file)
    all_interaction_groupby_user = all_interaction.groupby('user')
    all_interaction_groupby_user_list = [list(all_interaction_groupby_user.get_group(x).item) for x in
                                         all_interaction_groupby_user.groups]
    numbers_set = set(range(item_num))
    all_interaction_groupby_user_negative_list = (list(numbers_set - set(x)) for x in all_interaction_groupby_user_list)    # 针对这个列表，遍历其中的每一个，在每一个中随机抽取二十个作为负样本列表，并组成一个新的负样本列表
    import random
    all_interaction_groupby_user_negative_sample_list = []
    for x in all_interaction_groupby_user_negative_list:
        all_interaction_groupby_user_negative_sample_list.append(random.sample(x, 10))

    train_file_groupby_user_content_list = []
    train_file_groupby_user_content_target_list = []
    special_index = []
    print("groupby user ing。。。")
    for index, interaction_item_list in tqdm(enumerate(train_file_groupby_user_list)):
        interaction_item_list_len = min(len(interaction_item_list), 10)
        if interaction_item_list_len < 2:
            print("==========")
            print(index)
            special_index.append(index)
            print("==========")
            continue
        interaction_item_list = interaction_item_list[:interaction_item_list_len]
        target_item = interaction_item_list[-1]
        interaction_item_list = interaction_item_list[:-1]
        title_list = [item_raw_data.iloc[int(id)].title for id in interaction_item_list]
        target_item_list = item_raw_data.iloc[target_item].title
        train_file_groupby_user_content_list.append(title_list)
        train_file_groupby_user_content_target_list.append(target_item_list)

    print(f"special_index is {len(special_index)}")
    neg_title_list = []
    print("negative sample list processing。。。")
    for index, interaction_item_list in enumerate(all_interaction_groupby_user_negative_sample_list):
        if index in special_index:
            print(index)
            continue
        neg_item = interaction_item_list[0]
        neg_item_title = item_raw_data.iloc[neg_item].title
        neg_title_list.append(neg_item_title)

    train_user_preference_list = []
    print("train_file_groupby_user_content_list processing。。。")
    for index, title_list in tqdm(enumerate(train_file_groupby_user_content_list)):
        try:
            train_user_preference_list.append('\",\"'.join(title_list))
        except TypeError:
            print(f"An error occurred. The value of title_list is {title_list}, and its type is {type(title_list)}")

    train_positive_sample_list = []
    print("train_positive_sample_list processing。。。")
    for index, title in tqdm(enumerate(train_user_preference_list)):
        item_content_dict = {}
        instruction = instruction
        item_content_dict['instruction'] = instruction
        llama2_input = f'User preference:\"{title}\"\n,Whether the user will like the target item \"{train_file_groupby_user_content_target_list[index]}\"?'
        item_content_dict['input'] = llama2_input
        item_content_dict['output'] = "Yes"
        train_positive_sample_list.append(item_content_dict)

    train_negative_sample_list = []
    print("train_negative_sample_list processing。。。")
    for index, title in tqdm(enumerate(train_user_preference_list)):
        item_content_dict = {}
        instruction = instruction
        item_content_dict['instruction'] = instruction
        llama2_input = f'User preference:\"{title}\"\n,Whether the user will like the target item \"{neg_title_list[index]}\"?'
        item_content_dict['input'] = llama2_input
        item_content_dict['output'] = "No"
        train_negative_sample_list.append(item_content_dict)

    train_sample_list = train_positive_sample_list + train_negative_sample_list
    random.shuffle(train_sample_list)
    save_json_file(train_sample_list, datasetdir_path)

def save_file(train_user_preference_list) -> None:
    import pickle
    with open('train_user_preference_list.pkl', 'wb') as f:
        pickle.dump(train_user_preference_list, f)

def save_json_file(train_sample_list, datasetdir_path) -> None:
    print(" save file ing。。。")
    import json
    train_sample_json_file = os.path.join(datasetdir_path, 'train_sample.json')
    with open(train_sample_json_file, 'w') as f:
        json.dump(train_sample_list, f, indent=4)
    print("train_sample.json has been saved")


def create_new_prompt():
    pass

if __name__ == '__main__':
    item_num = 83046
    datasetdir_path = './Amazon/data/MusicalInstruements'
    item_raw_data_filename = 'item.csv'
    instruction = "Given the user's interaction item set, determine whether the user will like the target item by answering \"Yes\" or \"No.\""
    # genFile4SFTLlama(item_num, datasetdir_path, item_raw_data_filename, instruction)
    print("Done")