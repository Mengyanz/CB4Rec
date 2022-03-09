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
        """Use NRMS model. 
        """      
        super(NeuralLinearTS, self).__init__(device, args, name)
        self.n_inference = 1
        self.gamma = self.args.gamma
        self.n_arms = len(self.cb_news)
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