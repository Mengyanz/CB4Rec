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

def read_data(args):
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

    if args.mode == 'train':
        print('loading train_sam')
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/train_sam_uid.pkl'), 'rb') as f:
            train_sam = pickle.load(f)
        print('loading valid_sam')
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/valid_sam_uid.pkl'), 'rb') as f:
            valid_sam = pickle.load(f)

        if args.filter_user:
            print('filtering')
            train_sam, valid_sam = filter_sam(train_sam, valid_sam)

        return nid2index, news_info, news_index, embedding_matrix, train_sam, valid_sam
    elif args.mode == 'test':
        pass
    elif args.mode == 'cb':
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/sorted_train_sam_uid.pkl'), 'rb') as f:
            sorted_train_sam = pickle.load(f)
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/sorted_valid_sam_uid.pkl'), 'rb') as f:
            sorted_valid_sam = pickle.load(f)

        return nid2index, news_info, news_index, embedding_matrix, sorted_train_sam, sorted_valid_sam


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
        # print('pos: ', pos)
        # for n in candidate_news:
        #     print(n)
        #     print(self.nid2index[n])
        if type(candidate_news[0]) is str:
            assert candidate_news[0].startswith('N') # nid
            candidate_news = self.news_index[[self.nid2index[n] for n in candidate_news]]
        else: # nindex
            candidate_news = self.news_index[[n for n in candidate_news]]
        his = [self.nid2index[n] for n in his] + [0] * (self.max_his_len - len(his))
        his = self.news_index[his]
        
        label = np.array(0)
        return candidate_news, his, label

class NewsDataset(Dataset):
    def __init__(self, news_index):
        self.news_index = news_index
        
    def __len__(self):
        return len(self.news_index)
    
    def __getitem__(self, idx):
        return self.news_index[idx]

class UserDataset(Dataset):
    def __init__(self, args,samples,news_vecs,nid2index):
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
        his = [self.nid2index[n] for n in his] + [0] * (self.max_his_len - len(his))
        his = self.news_vecs[his]
        return his, tsp