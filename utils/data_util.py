"""Define utils for data. """

import os 
import numpy as np 
import pickle 
from pathlib import Path
import time
import datetime
from tqdm import tqdm
from collections import defaultdict, Counter
import copy
import random
import re
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import pickle
from datetime import datetime 
import math
# import uncertainty_toolbox as utc

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

def read_data(args, mode = 'train'):
    """Preprocessed data. 
    Args: 
        args: = parse_args(), where `parse_args()` from `configs`

    Return:
        nid2index: dict, maps each news id to a unique integer. 
        train_sam: list of (poss, negs, his, uid, tsp)
                poss: a list of strs, postive news ids 
                negs: a list of strs, negative news ids 
                his: a list of strings, history of browed news ids 
                uid: str, a user id 
                tsp: str, timestamp 
        val_sam: similar to train_sam


    @TODO: MIND data split into: train1, train2, and val. A simulator is trained on train1+train2 and is selected via val. 
    CB learner is pretrained only on train1. 
    """
    print('loading nid2index')
    with open(os.path.join(args.root_data_dir, args.dataset,  'utils', 'nid2index.pkl'), 'rb') as f:
        nid2index = pickle.load(f)
    print('loading news_info')
    with open(os.path.join(args.root_data_dir, args.dataset,  'utils', 'news_info.pkl'), 'rb') as f:
        news_info = pickle.load(f)
    print('loading embedding')
    embedding_matrix = np.load(os.path.join(args.root_data_dir, args.dataset,  'utils', 'embedding.npy'))
    print('loading news_index')
    news_index = np.load(os.path.join(args.root_data_dir, args.dataset,  'utils', 'news_index.npy'))

    if mode == 'train':
        print('loading train_sam')
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/train_contexts.pkl'), 'rb') as f:
            train_sam = pickle.load(f)
        print('loading valid_sam')
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/valid_contexts.pkl'), 'rb') as f:
            valid_sam = pickle.load(f)

        return nid2index, news_info, news_index, embedding_matrix, train_sam, valid_sam
    elif mode == 'test':
        pass
    elif mode == 'cb':
        # with open(os.path.join(args.root_data_dir, args.dataset, 'utils/sorted_train_sam_uid.pkl'), 'rb') as f:
        #     sorted_train_sam = pickle.load(f)
        # with open(os.path.join(args.root_data_dir, args.dataset, 'utils/sorted_valid_sam_uid.pkl'), 'rb') as f:
        #     sorted_valid_sam = pickle.load(f)

        cb_users = np.load(os.path.join(args.root_data_dir, args.dataset,  'cb_users.npy'),  allow_pickle=True)
        # cb_news = np.load(os.path.join(args.root_data_dir, args.dataset,  'cb_news.npy'), allow_pickle=True)[0]
        with open(os.path.join(args.root_data_dir, args.dataset,  'cb_news.pkl'), 'rb') as f:
            cb_news = pickle.load(f)

        return nid2index, news_info, news_index, embedding_matrix, cb_users, cb_news

def load_word2vec(args, utils = 'utils'): 
    """Load word2vec and nid2index
    """
    print('loading nid2index')
    with open(os.path.join(args.root_data_dir, args.dataset,  utils, 'nid2index.pkl'), 'rb') as f:
        nid2index = pickle.load(f)

    # print('loading news_info')
    # with open(os.path.join(args.root_data_dir, args.dataset,  'utils', 'news_info.pkl'), 'rb') as f:
    #     news_info = pickle.load(f)

    print('loading word2vec')
    word2vec = np.load(os.path.join(args.root_data_dir, args.dataset,  utils, 'embedding.npy'))

    print('loading nindex2vec')
    news_index = np.load(os.path.join(args.root_data_dir, args.dataset,  utils, 'news_index.npy'))

    return nid2index, word2vec, news_index


def load_sim_data(args):
    """Load data for training and evaluating a simulator. """

    print('loading train_contexts')
    with open(os.path.join(args.root_data_dir, args.dataset,  'utils', 'train_contexts.pkl'), 'rb') as f:
        train_contexts = pickle.load(f)     

    print('loading valid_contexts')
    with open(os.path.join(args.root_data_dir, args.dataset,  'utils', 'valid_contexts.pkl'), 'rb') as f:
        valid_contexts = pickle.load(f)   
    
    return train_contexts, valid_contexts

def load_cb_train_data(args, trial=0):
    with open(os.path.join(args.root_data_dir, args.dataset,  'utils', \
        'cb_train_contexts_nuser={}_splitratio={}_trial={}.pkl'.format(args.num_selected_users, args.cb_train_ratio, trial)), 'rb') as f:
        cb_train_contexts = pickle.load(f)  

    return cb_train_contexts

def load_cb_valid_data(args, trial=0):
    with open(os.path.join(args.root_data_dir, args.dataset,  'utils', \
        'cb_valid_contexts_nuser={}_splitratio={}_trial={}.pkl'.format(args.num_selected_users, args.cb_train_ratio, trial)), 'rb') as f:
        cb_valid_contexts = pickle.load(f)  

    return cb_valid_contexts

def load_cb_topic_news(args):
    fname = os.path.join(args.root_data_dir, "large/utils/cb_news.pkl") 
    with open(fname, 'rb') as f: 
        cb_news = pickle.load(f)
    return cb_news 

def newsample(nnn, ratio):
    if ratio > len(nnn):
        return nnn + ["<unk>"] * (ratio - len(nnn))
    else:
        return random.sample(nnn, ratio)

class TrainDataset(Dataset):
    def __init__(self, args, samples, nid2index, news_index):
        self.news_index = news_index
        self.nid2index = nid2index
        self.samples = samples
        self.npratio = args.npratio 
        self.max_his_len = args.max_his_len
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # pos, neg, his, neg_his
        pos, neg, his, uid, tsp = self.samples[idx]
        neg = newsample(neg, self.npratio)
        
        candidate_news = [pos] + neg

        candidate_news_vecs = []
        for n in candidate_news:
            if type(n) is str:
                candidate_news_vecs.append(self.news_index[self.nid2index[n]])
            else:
                candidate_news_vecs.append(self.news_index[n])
            
        if len(his) > self.max_his_len: 
            his = random.sample(his, self.max_his_len)

        if len(his) > 0:
            if type(his[0]) is str:
                his = [self.nid2index[n] for n in his] 
            else:
                his = his
        his = self.news_index[his + [0] * (self.max_his_len - len(his))]
        
        label = np.array(0)
        return np.array(candidate_news_vecs), his, label

class SimTrainDataset(Dataset):
    def __init__(self, args, nid2index, nindex2vec, samples):
        self.nid2index = nid2index 
        self.nindex2vec = nindex2vec 
        self.samples = samples 
        self.npratio = args.sim_npratio 
        self.max_his_len = args.max_his_len 

    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, idx):
        pos, neg, his, uid, tsp = self.samples[idx]
        neg = newsample(neg, self.npratio)
        candidate_news = [pos] + neg
        assert type(candidate_news[0]) is str 
        candidate_news = self.nindex2vec[[self.nid2index[n] for n in candidate_news]] 

        if len(his) > self.max_his_len: 
            his = random.sample(his, self.max_his_len)

        his = self.nindex2vec[ [self.nid2index[n] for n in his] + [0]*(self.max_his_len - len(his)) ]
        label = np.zeros(1 + self.npratio, dtype=float)
        label[0] = 1 
        return candidate_news, his, torch.Tensor(label)  

class SimTrainDatasetPropensity(Dataset):
    """Add mask and inverse propensity weight to handle imbalanced class. 
    """
    def __init__(self, args, nid2index, nindex2vec, samples):
        self.nid2index = nid2index 
        self.nindex2vec = nindex2vec 
        self.samples = samples 
        self.npratio = args.sim_npratio 
        self.max_his_len = args.max_his_len 

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # pos, neg, his, neg_his
        pos, neg, his, uid, tsp = self.samples[idx]
        neg = newsample(neg, self.npratio)
        
        candidate_news = [pos] + neg
        assert type(candidate_news[0]) is str 
        candidate_news = self.nindex2vec[[self.nid2index[n] for n in candidate_news]] 

        if len(his) > self.max_his_len: 
            his = random.sample(his, self.max_his_len)

        his = self.nindex2vec[ [self.nid2index[n] for n in his] + [0]*(self.max_his_len - len(his)) ]
        label = np.zeros(1 + self.npratio, dtype=float)
        label[0] = 1 
        return candidate_news, his, torch.Tensor(label)  

class SimValDataset(Dataset):
    def __init__(self, args, nid2index, nindex2vec, samples):
        self.nid2index = nid2index 
        self.nindex2vec = nindex2vec 
        self.samples = samples 
        self.max_his_len = args.max_his_len 

    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, idx):
        nidx, labels, his, uid, tsp = self.samples[idx]
        nvecs = self.nindex2vec[[self.nid2index[nid] for nid in nidx]]

        if len(his) > self.max_his_len: 
            his = random.sample(his, self.max_his_len)

        his = self.nindex2vec[ [self.nid2index[n] for n in his] + [0]*(self.max_his_len - len(his)) ]
        labels = torch.Tensor(np.array(labels).astype('float32'))
        return nvecs, his, labels  


class SimEvalDataset(Dataset):
    def __init__(self, args, uids, nindex2vec, clicked_history): 
        self.nindex2vec = nindex2vec 
        self.uids = uids 
        # self.candidate_news = self.nindex2vec[[n for n in news_indexes]]
        self.clicked_history = clicked_history 
        self.max_his_len = args.max_his_len 

    def __len__(self):
        return len(self.uids)

    def __getitem__(self,idx): 
        hist = self.clicked_history[self.uids[idx]]
        hist = hist + [0] * (self.max_his_len - len(hist))
        hist = self.nindex2vec[hist]
        return hist 

class SimEvalDataset2(Dataset):
    def __init__(self, args, cand_news, nindex2vec): 
        self.cand_news = cand_news 
        self.nindex2vec = nindex2vec 

    def __len__(self):
        return len(self.cand_news)

    def __getitem__(self,idx): 
        return self.nindex2vec[self.cand_news[idx]] 

class NewsDataset(Dataset):
    def __init__(self, news_index):
        self.news_index = news_index
        
    def __len__(self):
        return len(self.news_index)
    
    def __getitem__(self, idx):
        return self.news_index[idx]

class NewsDataset2(Dataset):
    def __init__(self, news2vec, news_indexes):
        self.news2vec = news2vec
        self.news_indexes = news_indexes
        
    def __len__(self):
        return len(self.news_indexes)
    
    def __getitem__(self, idx):
        return self.news2vec[self.news_indexes[idx]]

class UserDataset(Dataset):
    def __init__(self, args, samples, news_vecs, nid2index):
        """
        Args:
            samples: list of (poss, negs, his, uid, tsp)
                poss: a list of strs, postive news ids 
                negs: a list of strs, negative news ids 
                his: a list of strings, history of browed news ids 
                uid: str, a user id 
                tsp: str, timestamp 
        """
                 
        self.samples = samples
        self.news_vecs = news_vecs
        self.nid2index = nid2index
        self.max_his_len = args.max_his_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        poss, negs, his, uid, tsp = self.samples[idx]
        his = [self.nid2index[n] for n in his] + [0] * (self.max_his_len - len(his)) #@TODO: handle the case len(his) > max_his_len 
        his = self.news_vecs[his] # (max_his_len, max_title_len)
        return his, tsp

class UserDataset2(Dataset):
    def __init__(self, args, nid2index , news2code_fn, clk_history):
        self.nid2index = nid2index 
        self.news2code_fn = news2code_fn 
        self.clk_history = clk_history
        self.max_his_len = args.max_his_len

    def __len__(self):
        return len(self.clk_history)

    def __getitem__(self, idx): 
        clk_hist = self.clk_history[idx] 

        #@TODO: handle the case len(his) > max_his_len 
        clk_hist = [self.nid2index[i] for i in clk_hist] + [0] * (self.max_his_len - len(clk_hist)) 
        # return self.news2code_fn(clk_hist)
        self.news2code_fn([0,1])
        return clk_hist