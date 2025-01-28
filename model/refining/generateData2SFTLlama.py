import os
from tqdm import tqdm
import pandas as pd
def saveUserPreference(item_num, datasetdir_path, item_raw_data_filename, easy_instruction, hard_instruction) -> None:
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
        all_interaction_groupby_user_negative_sample_list.append(random.sample(x, 5))

    train_file_groupby_user_content_list = []
    train_file_groupby_user_content_target_list = []
    train_user_index_list = []
    special_index = []
    print("groupby user ing。。。")
    for index, interaction_item_list in tqdm(enumerate(train_file_groupby_user_list)):
        interaction_item_list_len = min(len(interaction_item_list), 5)
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
        paired_list = zip(interaction_item_list, title_list)
        formatted_list = [f"item_id: {id}, item_title: {title}" for id, title in paired_list]


        target_item_list = item_raw_data.iloc[target_item].title

        target_item_pair = f"item_id: {target_item}, item_title: {target_item_list}"
        train_file_groupby_user_content_list.append(formatted_list)
        train_file_groupby_user_content_target_list.append(target_item_pair)
        train_user_index_list.append(index)

    print(f"special_index is {len(special_index)}")
    neg_title_list = []
    print("negative sample list processing。。。")
    for index, interaction_item_list in enumerate(all_interaction_groupby_user_negative_sample_list):
        if index in special_index:
            print(index)
            continue
        neg_item = interaction_item_list[0]
        neg_item_title = item_raw_data.iloc[neg_item].title
        neg_item_pair = f"item_id: {neg_item}, item_title: {neg_item_title}"
        neg_title_list.append(neg_item_pair)

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
        instruction = easy_instruction
        item_content_dict['instruction'] = instruction
        llama2_input = f'User preference: user_id: {train_user_index_list[index]},\"{title}\"\n,Whether the user will like the target item \"{train_file_groupby_user_content_target_list[index]}\"?'
        item_content_dict['input'] = llama2_input
        item_content_dict['output'] = "Yes"
        train_positive_sample_list.append(item_content_dict)

    train_negative_sample_list = []
    print("train_negative_sample_list processing。。。")
    for index, title in tqdm(enumerate(train_user_preference_list)):
        item_content_dict = {}
        instruction = easy_instruction
        item_content_dict['instruction'] = instruction
        llama2_input = f'User preference: user_id: {train_user_index_list[index]},\"{title}\"\n,Whether the user will like the target item \"{neg_title_list[index]}\"?'
        item_content_dict['input'] = llama2_input
        item_content_dict['output'] = "No"
        train_negative_sample_list.append(item_content_dict)

    train_sample_list = train_positive_sample_list + train_negative_sample_list
    random.shuffle(train_sample_list)
    hard_prompt_target_item_index = []
    for index, neg_interaction_item_list in enumerate(all_interaction_groupby_user_negative_sample_list):
        if index in special_index:
            print(index)
            continue
        target_item = random.choice(all_interaction_groupby_user_list[index])
        hard_prompt_target_item_index.append(target_item)
        neg_interaction_item_list.append(target_item)
        random.shuffle(neg_interaction_item_list)

    hard_prompt_train_file_groupby_user_content_list = []
    for index, interaction_item_list in tqdm(enumerate(all_interaction_groupby_user_negative_sample_list)):
        if index in special_index:
            print(index)
            continue
        title_list = [item_raw_data.iloc[int(id)].title for id in interaction_item_list]
        paired_list = zip(interaction_item_list, title_list)
        formatted_list = [f"item_id: {id}, item_title: {title}" for id, title in paired_list]
        hard_prompt_train_file_groupby_user_content_list.append(formatted_list)

    train_hard_prompt_list = []
    for index, candidates in tqdm(enumerate(hard_prompt_train_file_groupby_user_content_list)):
        item_content_dict = {}
        instruction = hard_instruction
        item_content_dict['instruction'] = instruction
        llama2_input = (f'User preference: user_id: {train_user_index_list[index]},\" user_historical_interaction_set:{train_user_preference_list[index]}\"\n.Candidate items:[ \"{candidates}\"]\n,')
        item_content_dict['input'] = llama2_input
        item_content_dict['output'] = str(hard_prompt_target_item_index[index])
        train_hard_prompt_list.append(item_content_dict)

    # 对train_hard_prompt_list进行shuffle
    import random
    random.shuffle(train_hard_prompt_list)
    train_sample_list = train_sample_list + train_hard_prompt_list

    save_json_file(train_sample_list, datasetdir_path)

def save_file(train_user_preference_list) -> None:
    import pickle
    with open('train_user_preference_list.pkl', 'wb') as f:
        pickle.dump(train_user_preference_list, f)

def save_json_file(train_sample_list, datasetdir_path) -> None:
    print(" save file ing。。。")
    import json
    train_sample_json_file = os.path.join(datasetdir_path, 'train_SFT_LLAMA_sample.json')
    with open(train_sample_json_file, 'w') as f:
        json.dump(train_sample_list, f, indent=4)
    print("train_sample.json has been saved")


def create_new_prompt():
    pass

if __name__ == '__main__':
    item_num = 16980
    datasetdir_path = './CiteULike'
    item_raw_data_filename = 'item.csv'
    easy_instruction = "Given the user's interaction item set, determine whether the user will like the target item by answering \"Yes\" or \"No.\""
    hard_instruction = "User preference information includes user_id and user_historical_interaction_set information, and this user have also interacted with one of the candidate items. However, the data is missing. Can you help me find the missing interaction and provide the item_id of that item?"
    saveUserPreference(item_num, datasetdir_path, item_raw_data_filename, easy_instruction, hard_instruction)
    # genFile4SFTLlama(item_num, datasetdir_path, item_raw_data_filename, instruction)
    print("Done")