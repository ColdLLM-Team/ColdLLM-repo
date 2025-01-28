import os
from tqdm import tqdm
import pandas as pd
import pickle


def genFile4SFTLlama(datasetdir_path, item_raw_data_filename) -> None:
    print(1)
    train_file_name = 'warm_emb.csv'
    train_file_path = os.path.join(datasetdir_path, train_file_name)
    train_file = pd.read_csv(train_file_path)
    print(2)
    train_file_groupby_user = train_file.groupby('user')
    print(3)
    train_file_groupby_user_list = []
    count = 0
    missed_user_list = []
    for x in train_file_groupby_user.groups:
        if x != count:
            print(x)
            missed_user_list.append(x)
            train_file_groupby_user_list.append(list(train_file_groupby_user.get_group(x).item))
            count = x + 1
            continue
        count += 1
        train_file_groupby_user_list.append(list(train_file_groupby_user.get_group(x).item))
    # train_file_groupby_user_list = [list(train_file_groupby_user.get_group(x).item) for x in
    #                                 train_file_groupby_user.groups]
    sortedMissedUserList = sorted(missed_user_list, reverse=True)
    print(3.3)
    item_raw_data_path = os.path.join(datasetdir_path, item_raw_data_filename)
    item_raw_data = pd.read_csv(item_raw_data_path)
    item_raw_data = item_raw_data.fillna('')

    train_file_groupby_user_content_list = []
    train_file_groupby_user_content_target_list = []
    special_index = []
    print("groupby user ing。。。")
    for index, interaction_item_list in tqdm(enumerate(train_file_groupby_user_list)):
        interaction_item_list_len = min(len(interaction_item_list), 20)
        interaction_item_list = interaction_item_list[:interaction_item_list_len]
        target_item = interaction_item_list[-1]
        interaction_item_list = interaction_item_list[:-1]
        title_list = [item_raw_data.iloc[int(id)].title for id in interaction_item_list]
        target_item_list = item_raw_data.iloc[target_item].title
        train_file_groupby_user_content_list.append(title_list)
        train_file_groupby_user_content_target_list.append(target_item_list)
    train_user_preference_list = []
    for index, title_list in enumerate(train_file_groupby_user_content_list):
        train_user_preference_list.append('\",\"'.join(title_list))
    for index in sortedMissedUserList:
        train_user_preference_list.insert(index, 'default info')
        print(len(train_user_preference_list))
    with open(os.path.join(datasetdir_path,'train_user_preference_list.pkl'), 'wb') as f:
        pickle.dump(train_user_preference_list, f)
    print("done")


if __name__ == '__main__':
    genFile4SFTLlama('./Amazon/data/baby', 'item.csv')