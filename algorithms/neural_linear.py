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
from algorithms.neural_greedy import SingleStageNeuralGreedy
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news,load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class NeuralLinearTS(SingleStageNeuralGreedy):
    def __init__(self,device, args, name='NeuralLinearTS'):
        """Use NRMS model (disjoint model for each item)
        """      
        super(NeuralLinearTS, self).__init__(device, args, name)
        self.n_inference = 1
        self.gamma = self.args.gamma
        self.n_arms = len(self.cb_news) + 1
        self.args = args
        self.latent_dim = self.args.latent_dim
        print('Debug: init NeuralLinearTS...')
        # Gaussian prior for each beta_i
        self.lambda_prior = self.args.lambda_prior
        self.mu = [
            np.zeros(self.latent_dim)
            for _ in range(self.n_arms)
        ]
        self.cov = [(1.0 / self.lambda_prior) * np.eye(self.latent_dim)
                    for _ in range(self.n_arms)]
        self.precision = [
            self.lambda_prior * np.eye(self.latent_dim)
            for _ in range(self.n_arms)
        ]

        # Inverse Gamma prior for each sigma2_i
        self.a0 = 2
        self.b0 = 2

        self.a = [self.a0 for _ in range(self.n_arms)]
        self.b = [self.b0 for _ in range(self.n_arms)]
        # self.cb_indexs = self._get_cb_news_index([item for sublist in list(self.cb_news.values()) for item in sublist])
        print('Debug: finish init NeuralLinearTS.')
        # # pre-generate news embeddings
        # self.news_vecss = []
        # for i in range(self.n_inference): 
        #     news_vecs = self._get_news_vecs() # (n,d)
        #     self.news_vecss.append(news_vecs) 

    # def _get_cb_news_index(self, cb_news):
    #     """Generate cb news vecs by inferencing model on cb news

    #     Args
    #         cb_news: list of cb news samples

    #     Return
    #         cb_indexs: list of indexs corresponding to the input cb_news
    #     """
    #     print('#cb news: ', len(cb_news))
    #     cb_indexs = []
    #     for l in cb_news:
    #         nid = l.strip('\n').split("\t")[0]
    #         cb_indexs.append(self.nid2index[nid])
    #     return np.array(cb_indexs)   

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
            print('update linear model!')
            for action_v in items:
                # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
                
                user_vec = self._get_user_embs(uid, 0) # (1,d)
                repeat_user_vec = np.repeat(user_vec, repeats = len(items), axis = 0) # (n,d)
                news_embs = self.news_embs[0][items] # (n,d)
                # TODO: do proper dim reduction
                z = np.concatenate([repeat_user_vec[:, : int(self.args.latent_dim/2)], news_embs[:,:int(self.args.latent_dim/2)]], axis = 1) # (n,2d)
                
                y = rewards

                # The algorithm could be improved with sequential formulas (cheaper)
                s = np.dot(z.T, z)

                # Some terms are removed as we assume prior mu_0 = 0.
                precision_a = s + self.lambda_prior * np.eye(self.latent_dim)
                cov_a = np.linalg.inv(precision_a)
                mu_a = np.dot(cov_a, np.dot(z.T, y))

                # Inverse Gamma posterior update
                a_post = self.a0 + z.shape[0] / 2.0
                b_upd = 0.5 * np.dot(y.T, y)
                b_upd -= 0.5 * np.dot(mu_a.T, np.dot(precision_a, mu_a))
                b_post = self.b0 + b_upd

                # Store new posterior distributions
                self.mu[action_v] = mu_a
                self.cov[action_v] = cov_a
                self.precision[action_v] = precision_a
                self.a[action_v] = a_post
                self.b[action_v] = b_post  


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

        if self.preinference_mode:
            user_vec = self._get_user_embs(uid, 0) # (1,d)
            repeat_user_vec = np.repeat(user_vec, repeats = len(cand_news), axis = 0) # (n,d)
            news_embs = self.news_embs[0][cand_news] # (n,d)
            # TODO: do proper dim reduction
            context = np.concatenate([repeat_user_vec[:, : int(self.args.latent_dim/2)], news_embs[:,:int(self.args.latent_dim/2)]], axis = 1) # (n,2d)
            # print('Debug context shape: ', context.shape) 
            
        """Samples beta's from posterior, and chooses best action accordingly."""


        # Sample sigma2, and beta conditional on sigma2
        sigma2_s = [
            self.b[nindex] * invgamma.rvs(self.a[nindex])
            for nindex in cand_news
        ]

        try:
            beta_s = [
                np.random.multivariate_normal(self.mu[nindex],
                                              sigma2_s[i] * self.cov[nindex])
                for i, nindex in enumerate(cand_news)
            ]
        except np.linalg.LinAlgError as e:
            # TODO: check whether we want to do sample this way
            # Sampling could fail if covariance is not positive definite
            print('Exception when sampling for {}.'.format(self.name))
            print('Details: {} | {}.'.format(e.message, e.args))
            d = self.latent_dim
            beta_s = [
                np.random.multivariate_normal(np.zeros((d)), np.eye(d))
                for _ in cand_news
            ]

        # Apply Thompson Sampling to last-layer representation
        vals = [
            np.dot(beta_s[i], context[i].T) for i in
            range(len(cand_news))
        ]
        # vals = np.dot(np.array(beta_s), context.T)
        return np.array(cand_news)[np.array(np.argsort(vals)[::-1][:m])]
            

    def sample_actions(self, uids): 
        """Choose an action given a context. 
        Args:
            uids: a list of str uIDs. 
        Return: 
            topics: (len(uids), `rec_batch_size`)
            items: (len(uids), `rec_batch_size`) 
            numbers of items? 
        """
        # all_scores = []
        # self.model.eval()
        # for i in range(self.n_inference): # @TODO: accelerate
        #     user_vecs = self._get_user_embs(self.news_vecss[i], user_samples) # (b,d)
        #     scores = self.news_vecss[i][self.cb_indexs] @ user_vecs.T # (n,b) 
        #     all_scores.append(scores) 
        
        # all_scores = np.array(all_scores) # (n_inference,n,b)
        # mu = np.mean(all_scores, axis=0) 
        # std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
        # ucb = mu + std # (n,b) 
        # sorted_ids = np.argsort(ucb, axis=0)[-self.rec_batch_size:,:] 
        # return self.cb_indexs[sorted_ids], np.empty(0)
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings

        cand_news = [self.nid2index[n] for n in self.cb_news]
        rec_items = self.item_rec(uids, cand_news, self.rec_batch_size)

        return np.empty(0), rec_items

class NeuralLinearUCB_UserDisjoint(SingleStageNeuralGreedy):
    def __init__(self,device, args, name='NeuralLinearUCB-UserDisjoint'):
        """Use NRMS model (disjoint model for each user)
        """      
        super(NeuralLinearUCB, self).__init__(device, args, name)
        self.n_inference = 1
        self.gamma = self.args.gamma
        self.args = args
        self.dim = 400 # self.args.latent_dim

        self.theta = {} # key: uid, value: theta_u
        # self.D = defaultdict(list) # key: uid, value: list of nindex of uid's interactions
        # self.c = defaultdict(list) # key: uid, value: list of labels of uid's interactions

        self.A = {}
        self.Ainv = {}
        self.b = {}

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
        score_budget = self.per_rec_score_budget * m
        if len(cand_news)>score_budget:
            print('Randomly sample {} candidates news out of candidate news ({})'.format(score_budget, len(cand_news)))
            cand_news = np.random.choice(cand_news, size=score_budget, replace=False).tolist()

        # ser_vec = self._get_user_embs(uid, 0) # (1,d)
        X = self.news_embs[0][cand_news] # (n,d)

        if uid not in self.A:       
            self.A[uid] = np.identity(n=self.dim)
            self.Ainv[uid] = np.linalg.inv(self.A[uid])
            self.b[uid] = np.zeros((self.dim)) 
            # TODO: try init with nrms user embedding?
            if self.pretrained_mode and len(self.clicked_history[uid]) > 0:
                self.update(topics = None, items=self.clicked_history[uid], rewards=np.ones((len(self.clicked_history[uid]),)), mode = 'item', uid = uid)
            else:
                # REVIEW: alternatively, we can init theta to zeros
                self.theta[uid] = np.random.rand(self.dim)
                # self.theta[uid] = np.zeros(self.dim)
                # print('No history of user {}, init theta randomly!'.format(uid))
            
        mean = X.dot(self.theta[uid]) # n_cand, 
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv[uid]).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    def sample_actions(self, uids): 
        """Choose an action given a context. 
        Args:
            uids: a list of str uIDs. 

        Return: 
            topics: (len(uids), `rec_batch_size`)
            items: (len(uids), `rec_batch_size`) 
            numbers of items? 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings

        cand_news = [self.nid2index[n] for n in self.cb_news]
        rec_items = self.item_rec(uids, cand_news, self.rec_batch_size)

        return np.empty(0), rec_items

    def reset(self):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        # self.D = defaultdict(list) 
        # self.c = defaultdict(list)
        self.theta = {}
        self.A = {}
        self.Ainv = {}
        self.b = {}

class LogisticRegression2(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression2, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
        self.linear2 = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x, z):
        outputs = torch.sigmoid(self.linear1(x) + self.linear2(z))
        return outputs

class NeuralGLMUCB_UserItemHybrid(SingleStageNeuralGreedy):
    def __init__(self,device, args, name='NeuralGLMUCB-UserItemHybrid'):
        """Use NRMS model with logistic regression (user item hybrid)
        \bm{x}_{i}^T \hat{\bm{\theta}}_x +  {\hat{\bm{\theta}}_z}^T \bm{z}_{u}+\alpha  \sqrt{\bm{x}_i^T A_t^{-1} \bm{x}_i},
        """      
        super(NeuralGLMUCB_UserItemHybrid, self).__init__(device, args, name)
        self.n_inference = 1
        self.gamma = self.args.gamma
        self.args = args
        # self.dim = 400 # self.args.latent_dim

        self.A =  np.identity(n=self.args.latent_dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.args.latent_dim))

        self.lr_model = LogisticRegression2(self.args.latent_dim, 1)
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
            tr_users.append(uid)
            if len(poss) > 0:
                tr_samples.extend(poss)
                tr_rewards.extend([1] * len(poss))
                tr_neg_len = int(min(1.5 * len(poss), len(negs)))
                tr_samples.extend(np.random.choice(negs, size = tr_neg_len, replace=False))
                tr_rewards.extend([0] * tr_neg_len)

            self.data_buffer_lr.remove(l)
        # print('Debug tr_samples: ', tr_samples)
        # print('Debug tr_rewards: ', tr_rewards)
        # print('Debug self.data_buffer: ', self.data_buffer)
        return tr_samples, tr_rewards, tr_users

    def train_lr(self, uid):
        optimizer = optim.Adam(self.lr_model.parameters(), lr=self.args.lr)
        ft_sam, ft_labels, ft_users = self.construct_trainable_samples_lr()
        x = self.news_embs[0][ft_sam] # n_tr, n_dim
        z = np.array([self._get_user_embs(uid, 0) for uid in ft_users])# (b,d)
        # print('Debug x shape: ', x.shape)
        # print('Debug z shape: ', z.shape)
        if len(ft_sam) > 0:
            self.lr_model.train()
            for epoch in range(self.args.epochs):
                preds = self.lr_model(torch.Tensor(x), torch.Tensor(z))
                # print('Debug preds: ', preds)
                # print('Debug labels: ', ft_labels)
                loss = self.criterion(preds, torch.Tensor(ft_labels))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
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
        score_budget = self.per_rec_score_budget * m
        if len(cand_news)>score_budget:
            print('Randomly sample {} candidates news out of candidate news ({})'.format(score_budget, len(cand_news)))
            cand_news = np.random.choice(cand_news, size=score_budget, replace=False).tolist()

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

    def sample_actions(self, uids): 
        """Choose an action given a context. 
        Args:
            uids: a list of str uIDs. 

        Return: 
            topics: (len(uids), `rec_batch_size`)
            items: (len(uids), `rec_batch_size`) 
            numbers of items? 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings

        cand_news = [self.nid2index[n] for n in self.cb_news]
        rec_items = self.item_rec(uids, cand_news, self.rec_batch_size)

        return np.empty(0), rec_items

    def reset(self):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        
        self.A =  np.identity(n=self.args.latent_dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.args.latent_dim))

        self.lr_model = LogisticRegression2(self.args.latent_dim, 1)
        self.data_buffer_lr = [] # for logistic regression

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

        self.A =  np.identity(n=self.args.latent_dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.args.latent_dim))

        self.lr_model = LogisticRegression3(self.args.latent_dim, self.args.latent_dim, 1)
        self.criterion = torch.nn.BCELoss()
        self.data_buffer_lr = [] # for logistic regression