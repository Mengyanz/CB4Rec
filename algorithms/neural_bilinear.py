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

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.neural_greedy import SingleStageNeuralGreedy
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from algorithms.neural_linear import NeuralGLMUCB_UserItemHybrid
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news,load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class LogisticRegression3(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(LogisticRegression3, self).__init__()
        self.bilinear = torch.nn.Bilinear(input_dim1, input_dim2, output_dim)
        
    def forward(self, x, z):
        outputs = torch.sigmoid(self.bilinear(x,z))
        return outputs

class NeuralBilinUCB_Hybrid(NeuralGLMUCB_UserItemHybrid):
    # TODO: change CI into bilinear form
    def __init__(self,device, args, name='NeuralBilinUCB_Hybrid'):
        """Use NRMS model with logistic regression (disjoint model for each user)
        """      
        super(NeuralGLMUCB_UserItemHybrid, self).__init__(device, args, name)
        self.n_inference = 1
        self.gamma = self.args.gamma
        self.args = args
        # self.dim = 400 # self.args.latent_dim
        self.args.latent_dim = 64

        # model 
        self.model = NRMS_Model(self.word2vec, news_embedding_dim=64).to(self.device)
        self.model.eval()

        self.A =  np.identity(n=self.args.latent_dim**2)
        self.Ainv = np.linalg.inv(self.A)

        self.lr_model = LogisticRegression3(self.args.latent_dim, self.args.latent_dim, 1)
        self.criterion = torch.nn.BCELoss()
        self.data_buffer_lr = [] # for logistic regression

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
        score_budget = self.per_rec_score_budget * m
        if len(cand_news)>score_budget:
            print('Randomly sample {} candidates news out of candidate news ({})'.format(score_budget, len(cand_news)))
            cand_news = np.random.choice(cand_news, size=score_budget, replace=False).tolist()

        z = self._get_user_embs(uid, 0) # (1,d)
        X = self.news_embs[0][cand_news] # (n,d)
        cands = np.array([np.outer(x,z) for x in X]).reshape(len(cand_news),-1)

        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model.eval()
        mean = self.lr_model.forward(torch.Tensor(X), torch.Tensor(np.repeat(z, len(cands), axis = 0))).detach().numpy().reshape(len(cands),)
        # CI = np.array([self.gamma * np.sqrt(cand.dot(self.Ainv).dot(cand.T)) for cand in cands])
        # CI = self.gamma * np.sqrt(cands.dot(self.Ainv).dot(cands.T))
        CI = np.array([self.gamma * np.sqrt(cand.dot(self.Ainv).dot(cand.T)) for cand in cands])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        print(len(nid_argmax))
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    def reset(self):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        self.args.latent_dim=64
        
        self.A =  np.identity(n=self.args.latent_dim**2)
        self.Ainv = np.linalg.inv(self.A)

        self.lr_model = LogisticRegression3(self.args.latent_dim, self.args.latent_dim, 1)
        self.data_buffer_lr = [] # for logistic regression