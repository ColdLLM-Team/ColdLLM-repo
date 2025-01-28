from datetime import datetime

import utils
import numpy as np
import torch
from tqdm import tqdm
# 创建一个logger


def random_topk(knum, datadir, dataset, usernum):

    # logger = logging.getLogger()
    # loggeraftersetting = utils.set_Logger(datadir + dataset + '/log/', 'randomTop', logger)
    """
    Randomly select top k items for each item
    """
    convert_dict = utils.read_convert_dict_file(datadir, dataset)
    colditemList = convert_dict['cold_item']
    topk_dict = {}
    print('Randomly select top k items for each item')
    for item in colditemList:
        topk_dict[item] = np.random.choice(usernum, knum, replace=False)
    utils.save_top_csv_file(topk_dict, datadir + dataset + '/randomTop'+str(knum)+'.csv')
    print('Randomly select top k items for each item done')


def colab_topk(knum, datadir, dataset):
    colab_item_embedding = np.load(
        "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\ALDI\\baby_item_emb.npy")
    colab_user_embedding = np.load(
        "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\ALDI\\baby_user_emb.npy")

    convert_dict = utils.read_convert_dict_file(datadir, dataset)
    colditemList = convert_dict['cold_item']
    import torch
    cold_item_set_dict = {}
    user_tensor = torch.from_numpy(colab_user_embedding[1])
    for i in colditemList:
        item_tensor = torch.from_numpy(colab_item_embedding[i])
        result = item_tensor @ user_tensor.t()
        top_k_values, top_k_indices = torch.topk(result, k=knum, largest=True)
        cold_item_set_dict[i] = top_k_indices
    utils.save_top_csv_file(cold_item_set_dict, datadir + dataset + '/colabTop'+str(knum)+'.csv')

def tower_topk(knum, datadir, dataset, datetime_str):
    llama_tower_user_embedding = torch.load(
        'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\model\\user_content_emb_tensor_200_ml-1m0410.pt')
    llama_tower_item_embedding = torch.load(
        'C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\model\item_content_emb_tensor_200_ml-1m0410.pt')
    cold_item_set_dict = {}
    user_tensor = llama_tower_user_embedding
    convert_dict = utils.read_convert_dict_file(datadir, dataset)
    colditemList = convert_dict['cold_item']
    print('开始计算')
    for i in tqdm(colditemList):
        item_tensor = llama_tower_item_embedding[i]
        result = item_tensor @ user_tensor.t()
        top_k_values, top_k_indices = torch.topk(result, k=knum, largest=True)
        cold_item_set_dict[i] = top_k_indices
    utils.save_top_csv_file(cold_item_set_dict, datadir + dataset + '/towerTop'+str(knum)+datetime_str+'.csv')

def wide_topk(knum, datadir, dataset,colab_item_embedding_np_path,colab_user_embedding_np_path,llama_tower_user_embedding_path,llama_tower_item_embedding_path):
    colab_item_embedding = np.load(
        colab_item_embedding_np_path)
    colab_user_embedding = np.load(
        colab_user_embedding_np_path)
    colab_user_tensor = torch.from_numpy(colab_user_embedding[1])
    colab_item_tensor = torch.from_numpy(colab_item_embedding)
    convert_dict = utils.read_convert_dict_file(datadir, dataset)
    colditemList = convert_dict['cold_item']
    llama_tower_user_embedding = torch.load(
        llama_tower_user_embedding_path)
    llama_tower_item_embedding = torch.load(
        llama_tower_item_embedding_path)
    concat_item_emb = torch.cat((colab_item_tensor, llama_tower_item_embedding), dim=1)

    concat_user_emb = torch.cat((colab_user_tensor, llama_tower_user_embedding), dim=1)
    concat_cold_item_set_dict = {}
    user_tensor = concat_user_emb
    for i in colditemList:
        item_tensor = concat_item_emb[i]
        result = item_tensor @ user_tensor.t()
        top_k_values, top_k_indices = torch.topk(result, k=knum, largest=True)
        concat_cold_item_set_dict[i] = top_k_indices
    utils.save_top_csv_file(concat_cold_item_set_dict, datadir + dataset + '/wideTop'+str(knum) +'.csv')


if __name__ == '__main__':
    from datetime import datetime
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    baby_datadir = './Amazon/data/'
    baby_dataset = 'baby'
    # usernum = 19445
    knum = 20
    baby_colab_user_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\ALDI\\baby_user_emb.npy"
    baby_colab_item_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\ALDI\\baby_item_emb.npy"
    baby_llama_tower_user_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\\user_content_emb_tensor_200_AmazonBaby.pt"
    baby_llama_tower_item_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\item_content_emb_tensor_200_AmazonBaby.pt"

    citeulike_colab_user_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\CiteULike\CiteULike_user_emb.npy"
    citeulike_colab_item_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\CiteULike\CiteULike_item_emb.npy"
    citeulike_llama_tower_user_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\\user_content_emb_tensor_200_AmazonBaby.pt"
    citeulike_llama_tower_item_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\item_content_emb_tensor_200_AmazonBaby.pt"

    citeulike_colab_user_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\ALDI\\baby_user_emb.npy"
    citeulike_colab_item_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\ALDI\\baby_item_emb.npy"
    citeulike_llama_tower_user_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\\user_content_emb_tensor_200_AmazonBaby.pt"
    citeulike_llama_tower_item_embedding = "C:\\Users\yzh93\Desktop\LLAMA_TOWER\pythonProject\data\Amazon\data\\baby\item_content_emb_tensor_200_AmazonBaby.pt"
    # random_topk(knum, datadir, dataset, usernum)
    # colab_topk(knum, datadir, dataset)
    # tower_topk(knum, datadir, dataset)
    # wide_topk(knum, datadir, dataset)

    ml_1m_tower_datadir = './ml-1m/'
    ml_1m_tower_dataset = ''

    tower_topk(knum, ml_1m_tower_datadir, ml_1m_tower_dataset, date_time_str)

