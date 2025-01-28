import pickle
import numpy as np
import pandas as pd



def saveUserInfo2JsonFile(dataDir, userPreferencePath, savename):
    user_info_path = dataDir + userPreferencePath
    with open(user_info_path, 'rb') as f:
        user_info = pickle.load(f)
    ml_1m_user_info = []
    for index, user_content in enumerate(user_info):
        user_info_dict = {}
        user_content_info = f'User info:\"{user_content}\"\n'
        user_info_dict['user_info'] = user_content_info
        user_info_dict['item_info'] = ' '
        user_info_dict['label'] = 0
        ml_1m_user_info.append(user_info_dict)
    saveJsonFile(ml_1m_user_info, dataDir + savename)



def saveItemInfo2JsonFile(dataDir, ItemPreferencePath, savePath):
    item_content_data = pd.read_csv(dataDir + ItemPreferencePath)
    ml_1m_all_item_info = []
    for index, item in item_content_data.iterrows():
        item_info_dict = {}
        item_info = item['title']
        item_info = f'Item info:\"{item_info}\"\n'
        item_info_dict['user_info'] = ' '
        item_info_dict['item_info'] = item_info
        item_info_dict['label'] = 0
        ml_1m_all_item_info.append(item_info_dict)
    saveJsonFile(ml_1m_all_item_info, dataDir + savePath )


def saveJsonFile(readFile, savePath):
    import json
    with open(savePath, 'w') as f:
        json.dump(readFile, f, indent=4)


if __name__ == '__main__':
    dataDir = './Amazon/data/baby/'
    userPreferencePath = 'train_user_preference_list.pkl'
    Usersavename = 'baby4LlamaTower_all_user.json'
    ItemPreferencePath = 'item.csv'
    ItemSavePath = 'baby4LlamaTower_all_item.json'
    saveUserInfo2JsonFile(dataDir, userPreferencePath, Usersavename)
    saveItemInfo2JsonFile(dataDir, ItemPreferencePath, ItemSavePath)