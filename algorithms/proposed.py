"""Define a neural linear. """

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
from algorithms.neural_greedy import NeuralGreedy, NeuralGreedy_NeuralGreedy
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from algorithms.lr_model import LogisticRegression, LogisticRegressionAddtive, LogisticRegressionBilinear
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, GLMTrainDataset, load_word2vec, load_cb_topic_news,load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset
from torch.utils.tensorboard import SummaryWriter
import os
from scipy.optimize import newton, root 

class Two_NeuralGLMAddUCB(NeuralGreedy_NeuralGreedy):
    def __init__(self, args, device, name='2_neuralglmadducb'):
        """Use NRMS model. 
        """
        super(Two_NeuralGLMAddUCB, self).__init__(args, device, name) 
        self.gamma = self.args.gamma
        self.dim = self.args.news_dim # topic and news has the same dim

        # item
        self.A =  np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))

        self.lr_model = LogisticRegressionAddtive(self.dim, 1)
        self.criterion = torch.nn.BCELoss()
        self.data_buffer_lr = [] # for logistic regression

        # topic lr
        self.A_topic =  np.identity(n=self.dim)
        self.Ainv_topic = np.linalg.inv(self.A)
        self.b_topic = np.zeros((self.dim))

        self.lr_model_topic = LogisticRegressionAddtive(self.dim, 1)

        self.criterion = torch.nn.BCELoss()
        self.topic_embs = []
    
    def getInv(self, old_Minv, nfv):
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
        # tmp_a = old_Minv.dot(nfv).dot(nfv.T).dot(old_Minv)
        tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
        new_Minv=old_Minv-tmp_a/tmp_b
        return new_Minv
    
    @torch.no_grad()
    def _get_topic_embs(self):
        self.topic_model.eval() # disable dropout
        self.topic_embs = self.topic_model.get_topic_embeddings_byindex(self.topic_order).cpu().numpy()
        
    def update_data_buffer(self, pos, neg, uid, t): 
        self.data_buffer.append([pos, neg, self.clicked_history[uid], uid, t])
        self.data_buffer_lr.append([pos, neg, self.clicked_history[uid], uid, t])

    def construct_trainable_samples_lr(self):
        """construct trainable samples which will be used in NRMS model training
        from self.h_contexts, self.h_actions, self.h_rewards
        """
        tr_samples = []
        tr_rewards = []
        tr_users = []
        # print('Debug self.data_buffer: ', self.data_buffer)

        for i, l in enumerate(self.data_buffer_lr):
            poss, negs, his, uid, tsp = l
            # tr_users.append(uid)
            if len(poss) > 0:
                tr_samples.extend(poss)
                tr_rewards.extend([1] * len(poss))
                tr_neg_len = int(min(1.5 * len(poss), len(negs)))
                tr_samples.extend(np.random.choice(negs, size = tr_neg_len, replace=False))
                tr_rewards.extend([0] * tr_neg_len)
                tr_users.extend([uid]*(len(poss) + tr_neg_len))

            # REVIEW:
            self.data_buffer_lr.remove(l)
        # print('Debug tr_samples: ', tr_samples)
        # print('Debug tr_rewards: ', tr_rewards)
        # print('Debug self.data_buffer: ', self.data_buffer)
        return tr_samples, tr_rewards, tr_users
    
    def train_lr(self, uid_input=None, mode='item'):
        optimizer = optim.Adam(self.lr_model.parameters(), lr=self.args.glm_lr)
        ft_sam, ft_labels, ft_users = self.construct_trainable_samples_lr()
        if mode == 'item':
            if len(ft_sam) > 0:
                x = self.news_embs[0][ft_sam] # n_tr, n_dim
                z = np.array([self._get_user_embs(uid, 0) for uid in ft_users])
                z = z.reshape(-1, z.shape[-1]) # n_tr, n_dim
                self.lr_model.train()
                for epoch in range(self.args.epochs):
                    preds = self.lr_model(torch.Tensor(x), torch.Tensor(z)).ravel()
                    # print('Debug labels: ', ft_labels)
                    loss = self.criterion(preds, torch.Tensor(ft_labels))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                    print('Skip update cb item GLM learner due to lack valid samples!')
        elif mode == 'topic':
            if len(ft_sam) > 0:
                topic_indexs = [self.nid2topicindex[self.index2nid[n]] for n in ft_sam]
                x = self.topic_embs[topic_indexs] # n_tr, n_dim
                z = np.array([self._get_topic_user_embs(uid, 0).detach().cpu().numpy() for uid in ft_users])
                print('Debug x: ', x)
                print('Debug z: ', z)
                z = z.reshape(-1, z.shape[-1]) # n_tr, n_dim
                self.lr_model_topic.train()
                for epoch in range(self.args.epochs):
                    preds = self.lr_model_topic(torch.Tensor(x), torch.Tensor(z)).ravel()
                    # print('Debug labels: ', ft_labels)
                    loss = self.criterion(preds, torch.Tensor(ft_labels))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                    print('Skip update cb topic GLM learner due to lack valid samples!')
        

    def update(self, topics, items, rewards, mode = 'item', uid = None):
        """Updates the posterior using linear bayesian regression formula.
        
        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`/'item-linear'
        """
        print('size(data_buffer): {}'.format(len(self.data_buffer)))

        if mode == 'item-linear':
            print('Update glmucb parameters for user {}!'.format(uid))

            # item
            x = self.news_embs[0][items].T # n_dim, n_hist
            # Update parameters
            self.A+=x.dot(x.T) # n_dim, n_dim
            self.b+=x.dot(rewards) # n_dim, 
            for i in x.T:
                self.Ainv=self.getInv(self.Ainv,i) # n_dim, n_dim
            self.train_lr(mode='item')

            # topic
            
            x = self.topic_embs[topics].T # n_dim, n_hist
            # Update parameters
            self.A_topic+=x.dot(x.T) # n_dim, n_dim
            self.b_topic+=x.dot(rewards) # n_dim, 
            for i in x.T:
                self.Ainv_topic=self.getInv(self.Ainv_topic,i) # n_dim, n_dim
            self.train_lr(mode='topic')
        elif mode == 'item' or mode == 'topic':
            self.train(mode)
        else:
            raise NotImplementedError


    def item_rec(self, uid, cand_news, m = 1): 
        """From neural_linear NeuralGLMAddUCB
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings[]

        z = self._get_user_embs(uid, 0) # (1,d)
        X = self.news_embs[0][cand_news] # (n,d)

        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model.eval()
        mean = self.lr_model.forward(torch.Tensor(X), torch.Tensor(z)).detach().numpy().reshape(X.shape[0],)
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 
    
    @torch.no_grad()
    def topic_rec(self, uid, m = 1): 
        """
        Args:
            uid: str, a user id 
            m: int, number of items to rec 
        Return: 
            list, containing m element, where each element is a list of cand news index inside a topic (topic can be newly formed if we dynamically form topics)
        """
        if len(self.news_embs) < 1:
            self._get_news_embs(topic=True) # init news embeddings[]
        if len(self.topic_embs) < 1:
            self._get_topic_embs()

        z = self._get_topic_user_embs(uid, 0).cpu().numpy() # (1,d)
        X = self.topic_model.get_topic_embeddings_byindex(self.topic_order).cpu().numpy()

        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model.eval()
        mean = self.lr_model.forward(torch.Tensor(X), torch.Tensor(z)).detach().numpy().reshape(X.shape[0],)
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        sorted_topic_indexs = np.argsort(ucb)[::-1].tolist() # (len(uids),)
        recs = self.topic_cand_news_prep(sorted_topic_indexs,m)
        return recs

    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Reset the CB learner to its initial state (do this for each experiment). """
        # TODO: save and reload
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        self.data_buffer_lr = [] # for logistic regression
        
        self.A =  np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))
        self.lr_model = LogisticRegressionAddtive(self.dim, 1)
        
        # topic lr
        self.A_topic =  np.identity(n=self.dim)
        self.Ainv_topic = np.linalg.inv(self.A)
        self.b_topic = np.zeros((self.dim))
        self.lr_model_topic = LogisticRegressionAddtive(self.dim, 1)
        

class Two_NeuralGBiLinUCB(Two_NeuralGLMAddUCB):
    def __init__(self, args, device, name='2_NeuralGBiLinUCB'):
        """Use Two stage NRMS model with logistic regression (disjoint model for each user)
        """      
        super(Two_NeuralGBiLinUCB, self).__init__(args, device, name)
        
        self.A =  np.identity(n=self.dim**2)
        self.Ainv = np.linalg.inv(self.A)
        self.lr_model = LogisticRegressionBilinear(self.dim, self.dim, 1)
        
        # topic 
        self.A_topic =  np.identity(n=self.dim**2)
        self.Ainv_topic = np.linalg.inv(self.A)
        self.lr_model_topic = LogisticRegressionBilinear(self.dim, self.dim, 1)
    
    def update(self, topics, items, rewards, mode = 'item', uid = None):
        """Updates the posterior using linear bayesian regression formula.
        
        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`/'item-linear'
        """
        print('size(data_buffer): {}'.format(len(self.data_buffer)))

        if mode == 'item-linear': 
            # item
            print('Update glmucb item parameters')
            X = self.news_embs[0][items] # n1, n_hist
            z = self._get_user_embs(uid, 0)[0] # 1,d2
            for x in X:
                vec = np.outer(x.T,z).reshape(-1,) # d1d2,

                # Update parameters
                self.A+=np.outer(vec, vec.T) # d1d2, d1d2                
                self.Ainv=self.getInv(self.Ainv,vec) # n_dim, n_dim
            self.train_lr(mode='item')

            # topic
            print('Update glmucb topic parameters')
            X_topic = self.topic_embs[topics] # n_dim, n_hist
            # Update parameters
            z_topic = self._get_topic_user_embs(uid, 0).detach().cpu().numpy().reshape(1,-1) # 1,d2
            print('Debug X_topic shape: ', X_topic.shape)
            print('Debug z_topic shape: ', z_topic.shape)
            for x in X_topic:
                vec = np.outer(x.T,z_topic).reshape(-1,) # d1d2,

                # Update parameters
                self.A_topic+=np.outer(vec, vec.T) # d1d2, d1d2                
                self.Ainv_topic=self.getInv(self.Ainv_topic,vec) # n_dim, n_dim
            self.train_lr(mode='topic')
        elif mode == 'item' or mode == 'topic':
            self.train(mode)
        else:
            raise NotImplementedError
        
    def item_rec(self, uid, cand_news, m = 1): 
        """From neural_bilinear NeuralGBiLinUCB
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
        z = self._get_user_embs(uid, 0).reshape(1,-1) # (1,d)
        X = self.news_embs[0][cand_news] # (n,d)
        cands = np.array([np.outer(x,z) for x in X]).reshape(len(cand_news),-1)
        t2 = datetime.datetime.now()
        print('Debug news,user inference and get cands:, ', t2-t1)
        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model.eval()
        mean = self.lr_model.forward(torch.Tensor(X), torch.Tensor(np.repeat(z, len(cands), axis = 0))).detach().numpy().reshape(len(cands),)
        t3 = datetime.datetime.now()
        print('Debug get ucb mean:, ', t3-t2)

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
    
    @torch.no_grad()
    def topic_rec(self, uid, m = 1): 
        """
        Args:
            uid: str, a user id 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs(topic=True) # init news embeddings[]
        if len(self.topic_embs) < 1:
            self._get_topic_embs()
            
        t1 = datetime.datetime.now()
        z = self._get_topic_user_embs(uid, 0).cpu().numpy().reshape(1,-1) # (1,d)
        X = self.topic_model.get_topic_embeddings_byindex(self.topic_order).cpu().numpy()

        cands = np.array([np.outer(x,z) for x in X]).reshape(X.shape[0],-1)
        t2 = datetime.datetime.now()
        print('Debug news,user inference and get cands:, ', t2-t1)
        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model_topic.eval()
        print('Debug torch.Tensor(X).shape: ', torch.Tensor(X).shape)
        print('Debug torch.Tensor(np.repeat(z, len(cands), axis = 0)).shape: ', torch.Tensor(np.repeat(z, len(cands), axis = 0)).shape)
        mean = self.lr_model_topic.forward(torch.Tensor(X), torch.Tensor(np.repeat(z, len(cands), axis = 0))).detach().numpy().reshape(len(cands),)
        t3 = datetime.datetime.now()
        print('Debug get ucb mean:, ', t3-t2)

        CI = []
        Ainv_topic = torch.unsqueeze(torch.Tensor(self.Ainv_topic).to(self.device),dim=0) # 1,4096,4096
        cands = torch.unsqueeze(torch.Tensor(np.array(cands)).to(self.device),dim=1) # 5000,1,4096
        CI = torch.bmm(torch.bmm(cands, Ainv_topic.expand(cands.shape[0],-1,-1)),torch.transpose(cands, 1,2)).ravel().cpu().numpy()

        t4 = datetime.datetime.now()
        print('Debug get ucb CI (on GPU):, ', t4-t3)
        ucb = mean + self.gamma * np.sqrt(CI) # n_cand, 

        sorted_topic_indexs = np.argsort(ucb)[::-1].tolist() # (len(uids),)
        recs = self.topic_cand_news_prep(sorted_topic_indexs,m)
        return recs
    
    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        self.data_buffer_lr = [] # for logistic regression
        
        self.A =  np.identity(n=self.dim**2)
        self.Ainv = np.linalg.inv(self.A)
        self.lr_model = LogisticRegressionBilinear(self.dim, self.dim, 1)
        
        
        self.A_topic =  np.identity(n=self.dim**2)
        self.Ainv_topic = np.linalg.inv(self.A)
        self.lr_model_topic = LogisticRegressionBilinear(self.dim, self.dim, 1)
