import os
from os.path import join
import re
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
from parse import parse_args
import pickle
args = parse_args()

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    # 更改test: cold/warm/all/val
    @property
    def coldtestDict(self):
        raise NotImplementedError
    
    @property
    def warmtestDict(self):
        raise NotImplementedError
    
    @property
    def alltestDict(self):
        raise NotImplementedError
    
    @property
    def valDict(self):
        raise NotImplementedError
    
    @property
    def trainPos(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getallItems(self, users):
        raise NotImplemented

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.path = path
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0

        all_file = path + '/all.txt'
        if args.train_set == 0:
            train_file = path + '/train0.txt'
        elif args.train_set == 1:
            # train_files = [filename for filename in os.listdir(path) if re.match(r'train\d+\.txt', filename)]
            train_file = path + '/train20.txt'
        elif args.train_set == 2:
            train_file = path + '/train.txt'
        # train_file = '/home/jyjiang/workplace/LightGCN-PyTorch-master/data/CiteUlike/train.txt'
        print(train_file)
        coldtest_file = path + '/cold_test.txt'
        warmtest_file = path + '/warm_test.txt'
        alltest_file = path + '/all_test.txt'
        val_file = path + '/val.txt'     

        self.path = path
        trainUniqueUsers, trainItem, trainUser= [], [], []
        allUser, allItem = [], []

        coldtestUniqueUsers, coldtestItem, coldtestUser = [], [], []
        warmtestUniqueUsers, warmtestItem, warmtestUser = [], [], []
        alltestUniqueUsers, alltestItem, alltestUser = [], [], []
        valUniqueUsers, valItem, valUser = [], [], []

        self.traindataSize = 0
        self.coldtestDataSize = 0
        self.warmtestDataSize = 0
        self.alltestDataSize = 0
        self.valDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    # items = [int(i) for i in l[1:]]
                    items = [int(i) for i in l[1:] if i.strip() != '']
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    if items:
                        self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(coldtest_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    # items = [int(i) for i in l[1:]]
                    items = [int(i) for i in l[1:] if i.strip() != '']
                    uid = int(l[0])
                    coldtestUniqueUsers.append(uid)
                    coldtestUser.extend([uid] * len(items))
                    coldtestItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.coldtestDataSize += len(items)
        self.coldtestUniqueUsers = np.array(coldtestUniqueUsers)
        self.coldtestUser = np.array(coldtestUser)
        self.coldtestItem = np.array(coldtestItem)

        with open(warmtest_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    warmtestUniqueUsers.append(uid)
                    warmtestUser.extend([uid] * len(items))
                    warmtestItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.warmtestDataSize += len(items)
        self.warmtestUniqueUsers = np.array(warmtestUniqueUsers)
        self.warmtestUser = np.array(warmtestUser)
        self.warmtestItem = np.array(warmtestItem)

        with open(alltest_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    alltestUniqueUsers.append(uid)
                    alltestUser.extend([uid] * len(items))
                    alltestItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.alltestDataSize += len(items)
        
        self.alltestUniqueUsers = np.array(alltestUniqueUsers)
        self.alltestUser = np.array(alltestUser)
        self.alltestItem = np.array(alltestItem)

        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    valUniqueUsers.append(uid)
                    valUser.extend([uid] * len(items))
                    valItem.extend(items)
                    # self.m_item = max(self.m_item, max(items))
                    # self.n_user = max(self.n_user, uid)
                    self.valDataSize += len(items)
        self.valUniqueUsers = np.array(valUniqueUsers)
        self.valUser = np.array(valUser)
        self.valItem = np.array(valItem)

        with open(all_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    # items = [int(i) for i in l[1:]]
                    items = [int(i) for i in l[1:] if i.strip() != '']
                    uid = int(l[0])
                    allUser.extend([uid] * len(items))
                    allItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    
        self.n_user += 1
        self.m_item += 1
        self.allUser = np.array(allUser)
        self.allItem = np.array(allItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.coldtestDataSize} interactions for cold testing")
        print(f"{self.warmtestDataSize} interactions for warm testing")
        print(f"{self.alltestDataSize} interactions for all testing")
        print(f"{world.dataset} Sparsity : {self.trainDataSize/ self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.ItemNet = csr_matrix((np.ones(len(self.allUser)), (self.allUser, self.allItem)),
                                  shape=(self.n_user, self.m_item))

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._trainPos = self.getUserPosItems(list(range(self.n_user)))
        self._allItems = self.getallItems(list(range(self.n_user)))

        self.__coldtestDict = self.__build_coldtest()
        self.__warmtestDict = self.__build_warmtest()
        self.__alltestDict = self.__build_alltest()
        self.__valDict = self.__build_val()

        print(f"{world.dataset} is ready to go")
    
    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def coldtestDict(self):
        return self.__coldtestDict
    
    @property
    def warmtestDict(self):
        return self.__warmtestDict
    
    @property
    def alltestDict(self):
        return self.__alltestDict    

    @property
    def valDict(self):
        return self.__valDict  
    
    @property
    def trainPos(self):
        return self._trainPos
    
    @property
    def allItems(self):
        return self._allItems
    
    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix") 
        if self.Graph is None:  
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')  
                print("successfully loaded...") 
                norm_adj = pre_adj_mat   
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R    
                adj_mat[self.n_users:, :self.n_users] = R.T  
                adj_mat = adj_mat.todok()    
                
                rowsum = np.array(adj_mat.sum(axis=1))      
                d_inv = np.power(rowsum, -0.5).flatten()    
                d_inv[np.isinf(d_inv)] = 0.                 
                d_mat = sp.diags(d_inv)                     
                
                norm_adj = d_mat.dot(adj_mat)              
                norm_adj = norm_adj.dot(d_mat)             
                norm_adj = norm_adj.tocsr()                
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)        

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)      
                self.Graph = self.Graph.coalesce().to(world.device)       
                print("don't split the matrix")
            graph_load = self.Graph.to("cpu")
            dict_path = os.path.join(self.path, 'graph.pkl')
            pickle.dump(graph_load, open(dict_path, 'wb'), protocol=4)
        return self.Graph

    def __build_coldtest(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.coldtestItem):
            user = self.coldtestUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def __build_warmtest(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.warmtestItem):
            user = self.warmtestUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def __build_alltest(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.alltestItem):
            user = self.alltestUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_val(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.valItem):
            user = self.valUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getallItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.ItemNet[user].nonzero()[1])
        return posItems
    

