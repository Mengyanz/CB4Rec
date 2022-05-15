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

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.neural_greedy import NeuralGreedy
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from algorithms.lr_model import LogisticRegression, LogisticRegressionAddtive
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, GLMTrainDataset, load_word2vec, load_cb_topic_news,load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset
from torch.utils.tensorboard import SummaryWriter
import os
from scipy.optimize import newton, root 


class NeuralLinUCB(NeuralGreedy):
    def __init__(self, args, device, name='NeuralLinUCB'):
        """Use NRMS model (disjoint model for each user)
        """      
        super(NeuralLinUCB, self).__init__(args, device, name)
        self.gamma = self.args.gamma
        self.dim = self.args.news_dim

        self.theta = {} # key: uid, value: theta_u
        self.A = {}
        self.Ainv = {}
        self.b = {}

        # debug glm
        # out_folder = os.path.join(args.result_path, 'runs') # store final results
        # if not os.path.exists(out_folder):
        #     os.makedirs(out_folder)

        # out_path = os.path.join(out_folder, args.algo_prefix)
        # self.writer = SummaryWriter(out_path) # https://pytorch.org/docs/stable/tensorboard.html
        
    def getInv(self, old_Minv, nfv):
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
        # tmp_a = old_Minv.dot(nfv).dot(nfv.T).dot(old_Minv)
        tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
        new_Minv=old_Minv-tmp_a/tmp_b
        return new_Minv

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
            x = self.news_embs[0][items].T # n_dim, n_hist
            # Update parameters
            print('Debug x shape: ', x.shape)
            print('Debug A shape: ', self.A[uid].shape)
            self.A[uid]+=x.dot(x.T) # n_dim, n_dim
            self.b[uid]+=x.dot(rewards) # n_dim, 
            for i in x.T:
                self.Ainv[uid]=self.getInv(self.Ainv[uid],i) # n_dim, n_dim
            self.theta[uid]=np.dot(self.Ainv[uid], self.b[uid]) # n_dim,


    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        # ser_vec = self._get_user_embs(uid, 0) # (1,d)
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings
        X = self.news_embs[0][cand_news] # (n,d)

        if uid not in self.A:       
            self.A[uid] = np.identity(n=self.dim)
            self.Ainv[uid] = np.linalg.inv(self.A[uid])
            self.b[uid] = np.zeros((self.dim)) 
            if len(self.clicked_history[uid]) > 0 and not self.args.random_init:
                self.update(topics = None, items=self.clicked_history[uid], rewards=np.ones((len(self.clicked_history[uid]),)), mode = 'item-linear', uid = uid)
            # if self.args.pretrained_mode:
            #     self.pretrain_glm_learner(uid)
            else:
                print('Debug init linear parameters theta randomly')
                self.theta[uid] = np.random.rand(self.dim)
                # self.theta[uid] = np.zeros(self.dim)
                # print('No history of user {}, init theta randomly!'.format(uid))
            
        mean = X.dot(self.theta[uid]) # n_cand, 
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv[uid]).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    # debug glm
    def predict(self, uid, cand_news):
        X = self.news_embs[0][cand_news] # (n,d)
        preds = X.dot(self.theta[uid])
        return preds

    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Save and Reset the CB learner to its initial state (do this for each trial/experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        # self.D = defaultdict(list) 
        # self.c = defaultdict(list)
        self.theta = {}
        self.A = {}
        self.Ainv = {}
        self.b = {}

class NeuralGLMUCB(NeuralLinUCB):
    def __init__(self, args, device, name='NeuralGLMUCB'):
        """Use NRMS model (disjoint model for each user)
        """      
        super(NeuralGLMUCB, self).__init__(args, device, name)
        self.name = name 
        self.lr_models = {} # key: user, value: LogisticRegression(self.dim, 1)
        self.criterion = torch.nn.BCELoss()
        self.data_buffer_lr = [] # for logistic regression

    def set_clicked_history(self, init_clicked_history):
        """
        Args:
            init_click_history: list of init clicked history nindexes
        """
        self.clicked_history = init_clicked_history
        for uid, pos in self.clicked_history.items():
            self.data_buffer_lr.append([pos, [], uid])
        self.init_data_buffer_lr_len = len(self.data_buffer_lr)

    def update_data_buffer(self, pos, neg, uid, t): 
        self.data_buffer.append([pos, neg, self.clicked_history[uid], uid, t])
        self.data_buffer_lr.append([pos, neg, uid])

    def update(self, topics, items, rewards, mode = 'item', uid = None):
        """Updates the posterior using linear bayesian regression formula.
        
        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`/'item-linear'
        """
        print('size of data_buffer: {}; data_buffer_lr: {}'.format(len(self.data_buffer), len(self.data_buffer_lr)))
        if mode == 'item':
            self.train() 
            if self.args.reset_buffer:
                 # REVIEW: only reset buffer ever update_period round
                self.data_buffer_lr = []

        if mode == 'item-linear':
            print('Update glmucb parameters for user {}!'.format(uid))
            x = self.news_embs[0][items].T # n_dim, n_hist
            # Update parameters
            self.A[uid]+=x.dot(x.T) # n_dim, n_dim
            for i in x.T:
                self.Ainv[uid]=self.getInv(self.Ainv[uid],i) # n_dim, n_dim
            self.train_lr(uid)
    
    def construct_trainable_samples_lr(self, tr_uid):
        """construct trainable samples which will be used in NRMS model training
        from self.h_contexts, self.h_actions, self.h_rewards
        """
        tr_samples = []
        tr_rewards = []
        # print('Debug self.data_buffer_lr: ', self.data_buffer_lr)

        for i, l in enumerate(self.data_buffer_lr):
            poss, negs, uid = l
            if uid == tr_uid and len(poss) > 0:
                tr_samples.extend(poss)
                tr_rewards.extend([1] * len(poss))
                tr_neg_len = int(min(1.5 * len(poss), len(negs)))
                tr_samples.extend(np.random.choice(negs, size = tr_neg_len, replace=False))
                tr_rewards.extend([0] * tr_neg_len)
            # tr_samples.extend(poss)
            # tr_rewards.extend([1]*len(poss))
            # tr_samples.extend(negs)
            # tr_rewards.extend([0]*len(negs))

        # print('Debug tr_samples: ', tr_samples)
        # print('Debug tr_rewards: ', tr_rewards)
        # print('Debug self.data_buffer: ', self.data_buffer)
        return np.array(tr_samples), np.array(tr_rewards)
        
    def train_lr(self, uid):
        optimizer = optim.Adam(self.lr_models[uid].parameters(), lr=self.args.glm_lr)
        ft_sam, ft_labels = self.construct_trainable_samples_lr(uid)
        if len(ft_sam) > 0:
            
            x = self.news_embs[0][ft_sam] # n_tr, n_dim
            self.lr_models[uid].train()
            # REVIEW: for epoch in range(self.args.epochs):
            preds = self.lr_models[uid](torch.Tensor(x)).ravel()
            # print('Debug preds: ', preds)
            # print('Debug labels: ', rewards)
            loss = self.criterion(preds, torch.Tensor(ft_labels))
            print('Debug for uid {} loss {} '.format(uid, loss))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # debug glm
            # self.writer.add_scalars('{} Training Loss'.format(uid),
            #         {'Training': loss}, 
            #         len(self.data_buffer_lr) - self.init_data_buffer_lr_len)

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
            self._get_news_embs() # init news embeddings
        X = self.news_embs[0][cand_news] # (n,d)

        if uid not in self.A:       
            self.A[uid] = np.identity(n=self.dim)
            self.Ainv[uid] = np.linalg.inv(self.A[uid])
            self.lr_models[uid] = LogisticRegression(self.dim, 1)
            if len(self.clicked_history[uid]) > 0 and not self.args.random_init:
                print('Debug init lr parameters by clicked history ({})'.format(self.init_data_buffer_lr_len))
                self.train_lr(uid) # update by data_buffer_lr
            
        self.lr_models[uid].eval()
        mean = self.lr_models[uid].forward(torch.Tensor(X)).detach().numpy().reshape(X.shape[0],) # n_cand, 
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv[uid]).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    # debug glm
    def predict(self, uid, cand_news):
        X = self.news_embs[0][cand_news] # (n,d)
        preds = self.lr_models[uid].forward(torch.Tensor(X)).detach().numpy().reshape(X.shape[0],) # n_cand, 
        return preds


    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] 
        self.data_buffer_lr = [] 
        # self.D = defaultdict(list) 
        # self.c = defaultdict(list)
        self.lr_models = {}
        self.A = {}
        self.Ainv = {}

class NeuralGLMAddUCB(NeuralGreedy):
    def __init__(self, args, device, name='neural_glmadducb'):
        """Use NRMS model with logistic regression (user item hybrid)
        \bm{x}_{i}^T \hat{\bm{\theta}}_x +  {\hat{\bm{\theta}}_z}^T \bm{z}_{u}+\alpha  \sqrt{\bm{x}_i^T A_t^{-1} \bm{x}_i},
        """      
        super(NeuralGLMAddUCB, self).__init__(args, device, name)
        self.gamma = self.args.gamma
        self.dim = self.args.news_dim

        self.A =  np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))

        self.lr_model = LogisticRegressionAddtive(self.dim, 1)
        self.criterion = torch.nn.BCELoss()
        self.data_buffer_lr = [] # for logistic regression

    def getInv(self, old_Minv, nfv):
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
        # tmp_a = old_Minv.dot(nfv).dot(nfv.T).dot(old_Minv)
        tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
        new_Minv=old_Minv-tmp_a/tmp_b
        return new_Minv

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

        # print('Debug tr_samples: ', tr_samples)
        # print('Debug tr_rewards: ', tr_rewards)
        # print('Debug self.data_buffer: ', self.data_buffer)
        return tr_samples, tr_rewards, tr_users

    def train_lr(self, uid_input=None):
        optimizer = optim.Adam(self.lr_model.parameters(), lr=self.args.glm_lr)
        ft_sam, ft_labels, ft_users = self.construct_trainable_samples_lr()
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
        
        # ft_sam = self.construct_trainable_samples()
        # if len(ft_sam) > 0:
        #     print('Updating the internal item GLM model of the bandit!')
        #     ft_ds = GLMTrainDataset(self.args, ft_sam, self.nid2index, self.nindex2vec)
        #     ft_dl = DataLoader(ft_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        #     ft_loader = tqdm(ft_dl)

        #     self.lr_model.train()
        #     for cnt, batch_sample in enumerate(ft_loader):
        #         candidate_news_index, label, uids = batch_sample
        #         x = self.news_embs[0][candidate_news_index] # n_tr, n_dim
        #         z = np.array([self._get_user_embs(uid, 0) for uid in uids])
        #         z = z.reshape(-1, z.shape[-1]) # n_tr, n_dim
                
        #         for epoch in range(self.args.epochs):
        #             preds = self.lr_model_topic(torch.Tensor(x), torch.Tensor(z)).ravel()
        #             # print('Debug labels: ', ft_labels)
        #             loss = self.criterion(preds, torch.Tensor(label))
                    
        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()
        #         # REVIEW:
        #         if self.args.reset_buffer:
        #             self.data_buffer = [] # reset data buffer
        #     else:
        #         print('Skip update cb item GLM learner due to lack valid samples!')

        
    def update(self, topics, items, rewards, mode = 'item', uid = None):
        """Updates the posterior using linear bayesian regression formula.
        
        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`/'item-linear'
        """
        print('size of data_buffer: {}; data_buffer_lr: {}'.format(len(self.data_buffer), len(self.data_buffer_lr)))
        if mode == 'item':
            self.train() 
            if self.args.reset_buffer:
                 # REVIEW: only reset buffer ever update_period round
                self.data_buffer_lr = []

        if mode == 'item-linear':
            print('Update linucb parameters for user {}!'.format(uid))
            x = self.news_embs[0][items].T # n_dim, n_hist
            # Update parameters
            self.A+=x.dot(x.T) # n_dim, n_dim
            self.b+=x.dot(rewards) # n_dim, 
            for i in x.T:
                self.Ainv=self.getInv(self.Ainv,i) # n_dim, n_dim

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

    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        
        self.A =  np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))

        self.lr_model = LogisticRegressionAddtive(self.dim, 1)
        self.data_buffer_lr = [] # for logistic regression