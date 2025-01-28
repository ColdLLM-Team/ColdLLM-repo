import pandas as pd
import numpy as np
import pickle


def genPrompt4TopFile(datadir, dataset, topfilepath, itemcontentclms, pklfilesavepath):
    datapath = datadir + dataset
    with open(datapath + '/convert_dict.pkl', 'rb') as f:
        overall_dict = pickle.load(f)

    item_content = pd.read_csv(datapath +'/item.csv')
    cold_item_content = pd.DataFrame(columns=itemcontentclms)
    for i in range(len(overall_dict['cold_item'])):
        cold_item_content.loc[i] = [overall_dict['cold_item'][i], item_content.iloc[overall_dict['cold_item'][i]]['title']]

    cold_user_list = pd.read_csv(topfilepath)

    cold_user_list = cold_user_list[['item_id', 'user_id']]

    cold_user_list_dict = cold_user_list.groupby('item_id')['user_id'].apply(list)
    with open(datapath + '/train_user_preference_list.pkl', 'rb') as f:
        train_user_preference_list = pickle.load(f)
    citeUlike_cold_prompt_list = constructPrompt(cold_user_list_dict, train_user_preference_list, item_content)
    savePickleFile(citeUlike_cold_prompt_list, datapath + pklfilesavepath)



def savePickleFile(readFile, savePath):
    with open(savePath, 'wb') as f:
        pickle.dump(readFile, f)

def constructPrompt(cold_user_list_dict, train_user_preference_list, item_content):
    from tqdm import tqdm
    citeUlike_cold_prompt_list = []
    for key, value in tqdm(cold_user_list_dict.items()):
        cold_item_content_id = key
        for j in value:
            cold_prompt = {}
            user_id = j
            prompt = \
                f'''Given the user's interaction item set, determine whether the user will like the target item by answering \"Yes\" or \"No.\"\nUser preference:\"{train_user_preference_list[user_id]}\"\n,Whether the user will like the target item \"{item_content.iloc[key]['title']}\"?'''
            cold_prompt['user_id'] = user_id
            cold_prompt['cold_item_content_id'] = cold_item_content_id
            cold_prompt['prompt'] = prompt
            # print(cold_prompt)
            citeUlike_cold_prompt_list.append(cold_prompt)
    return citeUlike_cold_prompt_list

if __name__ == '__main__':
    babydatadir = './Amazon/data/'
    dataset = 'baby'
    baby_randomtopfilepath = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\\randomTop20.csv'
    baby_colabrandomtopfilepath = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\colabTop20.csv'
    baby_towertopfilepath = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\\towerTop20.csv'
    baby_widetowertopfilepath = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\wideTop20.csv'

    itemcontentclms = ['item_id', 'title']

    baby_randompklfilesavepath = 'ramdom_baby_cold_prompt_list.pkl'
    baby_colabpklfilesavepath = 'colab_baby_cold_prompt_list.pkl'
    baby_towerpklfilesavepath = 'tower_baby_cold_prompt_list.pkl'
    baby_widetowerpklfilesavepath = 'widetower_baby_cold_prompt_list.pkl'

    citeulikedir = '';
    citeulikedataset = 'CiteULike'
    citeulike_top10filepath = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\CiteULike\\top10_0120.csv'
    citeulike_top20filepath = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\CiteULike\\top20_0120.csv'
    citeulike_top50filepath = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\CiteULike\\top50_0120.csv'
    citeulike_top100filepath = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\CiteULike\\top100_0120.csv'

    itemcontentclms = ['item_id', 'title']

    mldatadir = './ml-1m/'
    dataset = ''
    ml_towerfilepath = 'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\ml-1m\ml1m-towerTop202024-04-10-15-10-30.csv'

    genPrompt4TopFile(mldatadir, dataset, ml_towerfilepath, itemcontentclms, 'ml_tower_cold_prompt_list_tower0410.pkl')

    # genPrompt4TopFile(babydatadir, dataset, baby_randomtopfilepath, itemcontentclms, baby_randompklfilesavepath)
    # genPrompt4TopFile(babydatadir, dataset, baby_colabrandomtopfilepath, itemcontentclms, baby_colabpklfilesavepath)
    # genPrompt4TopFile(babydatadir, dataset, baby_towertopfilepath, itemcontentclms, baby_towerpklfilesavepath)
    # genPrompt4TopFile(babydatadir, dataset, baby_widetowertopfilepath, itemcontentclms, baby_widetowerpklfilesavepath)
