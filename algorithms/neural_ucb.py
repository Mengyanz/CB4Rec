"""Define a simple UCB. """

import math 
import numpy as np 
import torch 
from torch.utils.data import DataLoader

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.nrms_model import NRMS_Model
from utils.data_util import read_data, NewsDataset, UserDataset

class SingleStageNeuralUCB(ContextualBanditLearner):
    def __init__(self,device, args, rec_batch_size = 1, n_inference=10, pretrained_mode=True, name='SingleStageNeuralUCB'):
        """Use NRMS model. 
            Args:
                rec_batch_size: int, recommendation size. 
                n_inference: int, number of Monte Carlo samples of prediction. 
                pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 

        """
        self.n_inference = n_inference 
        super(SingleStageNeuralUCB, self).__init__(rec_batch_size)

        self.pretrained_mode = pretrained_mode 
        self.name = name 
        self.device = device 
        self.args = args 

        # preprocessed data 
        self.nid2index, _, self.news_index, embedding_matrix, self.train_samples, self.valid_samples = read_data(args) 
        # self.news_index: (None, 30) - a set of integers. 

        self.num_news = self.news_index.shape[0] 

        # model 
        self.model = NRMS_Model(embedding_matrix).to(self.device)
        if self.pretrained_mode == 'pretrained':
            self.model.load_state_dict(torch.load(args.sim_path)) 

    def _get_news_vecs(self):
        news_dataset = NewsDataset(self.news_index) 
        news_dl = DataLoader(news_dataset,batch_size=1024, shuffle=False, num_workers=2)
        news_vecs = []
        for news in news_dl: # @TODO: avoid for loop
            news = news.to(self.device)
            news_vec = self.model.text_encoder(news).detach().cpu().numpy()
            news_vecs.append(news_vec)

        return np.concatenate(news_vecs) # (130381, 400)

    def _get_user_vecs(self, news_vecs, user_samples): 
        """Transform user_samples into representation vectors. 

        Args:
            user_samples: a list of (poss, negs, his, uid, tsp) 

        Return: 
            user_vecs: [None, dim]
        """
        user_dataset = UserDataset(self.args, user_samples, news_vecs, self.nid2index)
        user_vecs = []
        user_dl = DataLoader(user_dataset, batch_size=min(1024, len(user_samples)), shuffle=False, num_workers=2)

        for his_tsp in user_dl:
            his, tsp = his_tsp
            his = his.to(self.device)
            user_vec = self.model.user_encoder(his).detach().cpu().numpy()
            user_vecs.append(user_vec)
            # print(tsp)
        return np.concatenate(user_vecs)

    def sample_actions(self, user_samples): 
        all_scores = []
        for _ in range(self.n_inference):
            news_vecs = self._get_news_vecs() # (n,d)
            user_vecs = self._get_user_vecs(news_vecs, user_samples) # (b,d)
            scores = news_vecs @ user_vecs.T # (n,b) 
            all_scores.append(scores) 
        
        all_scores = np.array(all_scores) # (n_inference,n,b)
        mu = np.mean(all_scores, axis=0) 
        std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
        ucb = mu + std # (n,b) 
        sorted_ids = np.argsort(ucb, axis=0)[-self.rec_batch_size:,:] 
        return sorted_ids


class TwoStageNeuralUCB(ContextualBanditLearner):
    pass 