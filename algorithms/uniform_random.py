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
    def __init__(self, args, device, name='UniformRandom'):
        """Uniform random recommend news to user.
        """
        super(UniformRandom, self).__init__(args, device, name)
        self.name = name 

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