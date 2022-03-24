"""Define a uniform random recommendation policy. """

import math 
import numpy as np 
from collections import defaultdict
import torch 
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.neural_greedy import SingleStageNeuralGreedy
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class UniformRandom(ContextualBanditLearner):
    def __init__(self,device, args, name='UniformRandom'):
        """Uniform random recommend news to user.
        """
        super(UniformRandom, self).__init__(args, name)
        self.name = name 
        self.device = device 

        self.nid2index, word2vec, self.nindex2vec = load_word2vec(args)

        topic_news = load_cb_topic_news(args) # dict, key: subvert; value: list nIDs 
        cb_news = []
        for k,v in topic_news.items():
            cb_news.append(l.strip('\n').split("\t")[0] for l in v) # get nIDs 
        self.cb_news = [item for sublist in cb_news for item in sublist]

    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        cand_news = self.create_cand_set(cand_news,m)
        rec_items = np.random.choice(cand_news, size=m, replace=False).tolist()
        return rec_items 

    def sample_actions(self, uid): 
        """Choose an action given a context. 
        Args:
            uids: one str uID. 

        Return: 
            topics: (len(uid), `rec_batch_size`)
            items: (len(uid), `rec_batch_size`) 
        """
   
        cand_news = [self.nid2index[n] for n in self.cb_news]
        rec_items = self.item_rec(uid, cand_news, self.rec_batch_size)

        return np.empty(0), rec_items