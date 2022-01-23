"""Define NRMS simulator. """

import math 
import numpy as np 
import torch 
from torch.utils.data import DataLoader

from core.simulator import Simulator 
from algorithms.nrms_model import NRMS_Model
from utils.data_util import read_data, NewsDataset, UserDataset

class NRMS_Sim(Simulator): 
    def __init__(self, device, args, pretrained_mode=True, name='NRMS_Simulator'): 
        """
        Args:
            pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 
        """
        self.pretrained_mode = pretrained_mode 
        self.name = name 
        self.device = device 
        self.args = args 

        # preprocessed data 
        self.nid2index, _, self.news_index, embedding_matrix, self.train_samples, self.valid_samples = read_data(args) 
        # self.news_index: (None, 30) - a set of integers. 


        # model 
        self.model = NRMS_Model(embedding_matrix).to(self.device)
        if self.pretrained_mode == 'pretrained':
            self.model.load_state_dict(torch.load(args.sim_path)) 


        # get news_vecs
        print('{}: Getting news_vecs'.format(self.name))
        self.news_dataset = NewsDataset(self.news_index) 
        news_dl = DataLoader(self.news_dataset,batch_size=1024, shuffle=False, num_workers=2)
        news_vecs = []
        for news in news_dl: # @TODO: avoid for loop
            news = news.to(self.device)
            news_vec = self.model.text_encoder(news).detach().cpu().numpy()
            news_vecs.append(news_vec)

        self.news_vecs = np.concatenate(news_vecs) # (130381, 400)


    def _get_user_vecs(self, user_samples): 
        """Transform user_samples into representation vectors. 

        Args:
            user_samples: a list of (poss, negs, his, uid, tsp) 

        Return: 
            user_vecs: [None, dim]
        """
        user_dataset = UserDataset(self.args, user_samples, self.news_vecs, self.nid2index)
        user_vecs = []
        user_dl = DataLoader(user_dataset, batch_size=min(1024, len(user_samples)), shuffle=False, num_workers=2)

        for his_tsp in user_dl:
            his, tsp = his_tsp
            his = his.to(self.device)
            user_vec = self.model.user_encoder(his).detach().cpu().numpy()
            user_vecs.append(user_vec)
            # print(tsp)
        return np.concatenate(user_vecs)

    
    def reward(self, user_samples, news_ids): 
        """Returns a simulated reward. 

        Args:
            user_samples: a list of m user samples
            news_id: a list of n int, news ids. 

        Return: 
            rewards: (m,n) of 0 or 1 
        """
        user_vecs = self._get_user_vecs(user_samples) 
        news_vecs = self.news_vecs[news_ids,:] 

        reward_scores = user_vecs @ news_vecs.T 
        # @TODO: convert scores into binary 
        return reward_scores 

