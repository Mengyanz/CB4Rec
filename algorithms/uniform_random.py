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
from algorithms.neural_greedy import NeuralGreedy
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class UniformRandom(ContextualBanditLearner):
    def __init__(self, args, device, name='UniformRandom'):
        """Uniform random recommend news to user.
        """
        super(UniformRandom, self).__init__(args, device, name)

    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        rec_items = np.random.choice(cand_news, size=m, replace=False).tolist()
        return rec_items 

class Random_Random(UniformRandom):
    def __init__(self, args, device, name='2_random'):
        """Randomly recommend topics and then randomly recommend news to user.
        """
        super(Random_Random, self).__init__(args, device, name)

    def topic_rec(self, uid, m = 1):
        rec_topics = np.random.choice(self.cb_topics, size=m, replace=False).tolist()
        return rec_topics

    def sample_actions(self, uid, cand_news = None):
        """Choose an action given a context. 
        
        Args:
            uids: a str uIDs (user id). 
            cand_news: list of candidate news indexes 
        Return: 
            topics: (len(uids), `rec_batch_size`)
            items: (len(uids), `rec_batch_size`) 
        """
        rec_topics = self.topic_rec(uid, m=self.rec_batch_size)
        rec_items = []
        for rec_topic in rec_topics:
            cand_news = [self.nid2index[n] for n in self.cb_news[rec_topic]]
            rec_item = self.item_rec(uid, cand_news, m=1)
            rec_items.append(rec_item[0])

        return rec_topics, rec_items