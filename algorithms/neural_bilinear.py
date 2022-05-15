"""Define a neural bilinear. """

import math 
import numpy as np 
import pandas as pd
from collections import defaultdict
import torch 
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import invgamma
import datetime

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.neural_greedy import NeuralGreedy
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from algorithms.neural_linear import NeuralGLMAddUCB
from algorithms.lr_model import LogisticRegressionBilinear
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news,load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class NeuralGBiLinUCB(NeuralGLMAddUCB):
    def __init__(self, args, device, name='NeuralGBiLinUCB'):
        """Use NRMS model with logistic regression (disjoint model for each user)
        """      
        super(NeuralGBiLinUCB, self).__init__(args, device, name)
        
        self.A =  np.identity(n=self.dim**2)
        self.Ainv = np.linalg.inv(self.A)
        self.lr_model = LogisticRegressionBilinear(self.dim, self.dim, 1)

    def update(self, topics, items, rewards, mode = 'item', uid = None):
        """Updates the posterior using linear bayesian regression formula.
        
        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`/'item-linear'
        """
        print('size(data_buffer): {}'.format(len(self.data_buffer)))
        if mode == 'item':
            self.train() 
            if self.args.reset_buffer:
                 # REVIEW: only reset buffer ever update_period round
                self.data_buffer_lr = []

        if mode == 'item-linear':
            print('Update linucb parameters for user {}!'.format(uid))
            X = self.news_embs[0][items] # n1, n_hist
            z = self._get_user_embs(uid, 0)[0] # 1,d2
            for x in X:
                vec = np.outer(x.T,z).reshape(-1,) # d1d2,

                # Update parameters
                self.A+=np.outer(vec, vec.T) # d1d2, d1d2                
                self.Ainv=self.getInv(self.Ainv,vec) # n_dim, n_dim
            self.train_lr(uid)

    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings[]
        t1 = datetime.datetime.now()
        z = self._get_user_embs(uid, 0) # (1,d)
        X = self.news_embs[0][cand_news] # (n,d)
        cands = np.array([np.outer(x,z) for x in X]).reshape(len(cand_news),-1)
        t2 = datetime.datetime.now()
        print('Debug news,user inference and get cands:, ', t2-t1)
        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model.eval()
        mean = self.lr_model.forward(torch.Tensor(X), torch.Tensor(np.repeat(z, len(cands), axis = 0))).detach().numpy().reshape(len(cands),)
        t3 = datetime.datetime.now()
        print('Debug get ucb mean:, ', t3-t2)
        # CI = np.array([self.gamma * np.sqrt(cand.dot(self.Ainv).dot(cand.T)) for cand in cands])
        # CI = self.gamma * np.sqrt(cands.dot(self.Ainv).dot(cands.T))
        # CI = np.array([self.gamma * np.sqrt(cand.dot(self.Ainv).dot(cand.T)) for cand in cands])
        # CI = np.array([self.gamma * np.sqrt(cand.dot(torch.Tensor(self.Ainv)).dot(cand.T)).detach().numpy() for torch.Tensor(cand) in cands])

        # CI = []
        # Ainv = torch.Tensor(self.Ainv).to(self.device)
        # for cand in cands:
        #     cand = torch.Tensor(cand).reshape(1,-1).to(self.device)
        #     temp = cand@Ainv@cand.T
        #     CI.append(self.gamma * np.sqrt(temp[0,0].cpu().numpy()))
        # CI = np.array(CI)

        CI = []
        Ainv = torch.unsqueeze(torch.Tensor(self.Ainv).to(self.device),dim=0) # 1,4096,4096
        cands = torch.unsqueeze(torch.Tensor(np.array(cands)).to(self.device),dim=1) # 5000,1,4096
        CI = torch.bmm(torch.bmm(cands, Ainv.expand(cands.shape[0],-1,-1)),torch.transpose(cands, 1,2)).ravel().cpu().numpy()

        t4 = datetime.datetime.now()
        print('Debug get ucb CI (on GPU):, ', t4-t3)
        ucb = mean + self.gamma * np.sqrt(CI) # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        
        self.A =  np.identity(n=self.dim**2)
        self.Ainv = np.linalg.inv(self.A)

        self.lr_model = LogisticRegressionBilinear(self.dim, self.dim, 1)
        self.data_buffer_lr = [] # for logistic regression