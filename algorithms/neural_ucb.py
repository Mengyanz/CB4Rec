"""Define a simple UCB. """

import math 
import numpy as np 
import pandas as pd
from collections import defaultdict
import torch 
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.neural_greedy import NeuralGreedy, Two_NeuralGreedy
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news,load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class NeuralDropoutUCB(NeuralGreedy):
    def __init__(self, args, device, name='NeuralDropoutUCB'):
        """Use NRMS model. 
        """      
        super(NeuralDropoutUCB, self).__init__(args, device, name)
        self.n_inference = self.args.n_inference 
        self.gamma = self.args.gamma

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

        all_scores = []           
        for i in range(self.n_inference): 
            user_vecs = self._get_user_embs(uid, i) # (b,d)
            scores = self.news_embs[i][cand_news] @ user_vecs.T # (n,b) 
            all_scores.append(scores) 
        
        all_scores = np.array(all_scores).squeeze(-1)  # (n_inference,n,b)
        mu = np.mean(all_scores, axis=0) 
        std = np.std(all_scores, axis=0) # / math.sqrt(self.n_inference) 
        
        ucb = mu + self.gamma * std # (n,) 
        nid_argmax = np.argsort(ucb, axis = 0)[::-1][:m].tolist() # (len(uids),)
        print('Debug mean: ', mu[np.array(nid_argmax)])
        print('Debug std: ', std[np.array(nid_argmax)])
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

class Two_NeuralDropoutUCB(Two_NeuralGreedy):  
    def __init__(self, args, device, name='2_neuralucb'):
        """Two stage exploration. Use NRMS model. 
            Args:
                rec_batch_size: int, recommendation size. 
                n_inference: int, number of Monte Carlo samples of prediction. 
                pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 
        """
        super(Two_NeuralDropoutUCB, self).__init__(args, device, name)    
        self.n_inference = args.n_inference  
        self.gamma = args.gamma

    @torch.no_grad()
    def topic_rec(self, uid, m=1):
        """
        Args:
            uid: str, a user id 
            m: int, number of items to rec 
        Return: 
            list, containing m element, where each element is a list of cand news index inside a topic (topic can be newly formed if we dynamically form topics)
        """

        if self.n_inference == 1:
            self.topic_model.eval() # disable dropout
        else:
            self.topic_model.train() # enable dropout, for dropout ucb
        if len(self.news_embs) < 1:
            self._get_news_embs(topic=True) # init news embeddings
            
        all_scores = []
        for i in range(self.args.n_inference):
            user_vector = self._get_topic_user_embs(uid, i) # reduction_dim
            topic_embeddings = self.topic_model.get_topic_embeddings_byindex(self.topic_order) # get all active topic scores, num x reduction_dim
            score = (topic_embeddings @ user_vector.unsqueeze(-1)).squeeze(-1).cpu().numpy() # num_topic
            all_scores.append(score)

        all_scores = np.array(all_scores) # n_inference, num_active_topic
        mu = np.mean(all_scores, axis=0) 
        std = np.std(all_scores, axis=0) # / math.sqrt(self.n_inference)
        # print('Debug topic std: ', std) 
        ucb = mu + self.gamma * std  # num_topic
        # for topic in self.active_topics:
        #     s = np.random.beta(a= self.alphas[topic], b= self.betas[topic])
        #     ss.append(s)
        sorted_topic_indexs = np.argsort(ucb)[::-1].tolist() # (len(uids),)
        recs = self.topic_cand_news_prep(sorted_topic_indexs,m)
        return recs  
    
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

        all_scores = []           
        for i in range(self.n_inference): 
            user_vecs = self._get_user_embs(uid, i) # (b,d)
            scores = self.news_embs[i][cand_news] @ user_vecs.T # (n,b) 
            all_scores.append(scores) 
        
        all_scores = np.array(all_scores).squeeze(-1)  # (n_inference,n,b)
        mu = np.mean(all_scores, axis=0) 
        std = np.std(all_scores, axis=0) # / math.sqrt(self.n_inference) 
        ucb = mu + self.gamma * std # (n,) 
        nid_argmax = np.argsort(ucb, axis = 0)[::-1][:m].tolist() # (len(uids),)
        print('Debug mean: ', mu[np.array(nid_argmax)])
        print('Debug std: ', std[np.array(nid_argmax)])
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 