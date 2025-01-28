import torch
import torch.nn.functional as F
from dataloader import BasicDataset
from torch import nn
import numpy as np
from parse import parse_args
args = parse_args()

import pickle
with open('/home/jyjiang/workplace/LightGCN-PyTorch-master/data/'+ str(args.dataset) + '/convert_dict.pkl', 'rb') as file:
    para_dict = pickle.load(file)

class BasicModel(nn.Module):    
    def __init__(self, config:dict, dataset:BasicDataset):
        super(BasicModel, self).__init__()
        self.config = config
        self.dataset = dataset
        self.f = nn.Sigmoid()
        self._init_weight()
    

    def _init_weight(self):
        raise NotImplementedError


    def computer(self):
        raise NotImplementedError


    def getUsersRating(self, users):
        user_emb, items_emb = self.computer()
        users_emb = user_emb[users.long()]
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)


    def getEmbedding(self, users, pos_items, neg_items):
        user_emb, item_emb = self.computer()
        users_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        users_emb_ego = users_emb
        pos_emb_ego = pos_emb 
        neg_emb_ego = neg_emb
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego


    def bpr_loss(self, users, pos, neg):
        user_emb, item_emb = self.computer()
        users_emb = user_emb[users.long()]
        pos_emb   = item_emb[pos.long()]
        neg_emb   = item_emb[neg.long()]
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss


    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

class basicModel(nn.Module):    
    def __init__(self):
        super(basicModel, self).__init__() 
    
    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(basicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class NGCF(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super().__init__(config, dataset)

    def _init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        
        initializer = nn.init.xavier_uniform_
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        initializer(self.embedding_user.weight)
        initializer(self.embedding_item.weight)

        
        for k in range(self.n_layers):
            # trans
            setattr(self, 'W_gc_%d'%k, nn.Parameter(initializer(torch.empty(self.latent_dim, self.latent_dim))))
            setattr(self, 'b_gc_%d'%k, nn.Parameter(initializer(torch.empty(1, self.latent_dim))))
            # bi message
            setattr(self, 'W_bi_%d'%k, nn.Parameter(initializer(torch.empty(self.latent_dim, self.latent_dim))))
            setattr(self, 'b_bi_%d'%k, nn.Parameter(initializer(torch.empty(1, self.latent_dim))))
    
        self.Graph = self.dataset.getSparseGraph()
        print(f"NGCF is already to go")
        
    def computer(self):
        """
        propagate methods for NGCF
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        ego_emb = torch.cat([users_emb, items_emb])
        embs = [ego_emb]
        Graph = self.Graph
        
        # propagation
        for k in range(self.n_layers):
            side_embeddings = torch.sparse.mm(Graph, ego_emb)
            
            # transformed sum messages of neighbors.
            # W1
            sum_embeddings = torch.matmul(side_embeddings, getattr(self, 'W_gc_%d'%k)) + getattr(self, 'b_gc_%d'%k)
            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_emb, side_embeddings)

            # transformed bi messages of neighbors.
            # W2
            bi_embeddings = torch.matmul(bi_embeddings, getattr(self, 'W_bi_%d'%k)) + getattr(self, 'b_bi_%d'%k)
            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            embs.append(norm_embeddings)
        
        # embs = torch.cat(embs, dim=1)   
        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)
        users, items = torch.split(embs, [self.num_users, self.num_items])
        return users, items
    