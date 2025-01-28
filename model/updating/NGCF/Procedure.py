import world
import numpy as np
import pandas as pd
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
from parse import parse_args
args = parse_args()

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)

    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks: 
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, i, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    Recmodel: model.LightGCN
    import pickle

    if i == 0:
        # test cold
        testDict: dict = dataset.coldtestDict
        testtype = 'cold'
    elif i == 1:
        # test warm
        testDict: dict = dataset.warmtestDict
        testtype = 'warm'
    elif i == 2:
        # test all
        testDict: dict = dataset.alltestDict
        testtype = 'all'
    else:
        testDict: dict = dataset.valDict
        testtype = 'val'
   
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())[:args.n_test_user]
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        warm_items = pd.read_csv('../data/' + world.dataset + '/item_sets.csv')['warm'].dropna().tolist()
        cold_items = pd.read_csv('../data/' + world.dataset + '/item_sets.csv')['cold'].dropna().tolist()
        if len(users) % u_batch_size != 0:
            total_batch = len(users) // u_batch_size + 1
        else:
            total_batch = len(users) // u_batch_size
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            allItem = dataset.getallItems(batch_users)
            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(allItem):
                items = np.array(list(set(items) - set(groundTrue[range_i])), np.int32)
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            if i == 0:
                rating[:, warm_items] = -(1<<10)
            elif i == 1:
                rating[:, cold_items] = -(1<<10)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        if i !=3:
            if world.tensorboard:
                w.add_scalars(f'Test_{testtype}/Recall@{world.topks}',
                            {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test_{testtype}/Precision@{world.topks}',
                            {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test_{testtype}/NDCG@{world.topks}',
                            {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            if multicore == 1:
                pool.close()
        print("{} recommendation result@{}: REC, NDCG: {:.4f}, {:.4f}".format(testtype, world.topks[1], results['recall'][1], results['ndcg'][1]))
        return results
