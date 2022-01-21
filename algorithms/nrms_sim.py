"""Define NRMS simulator. """

import math 
import numpy as np 
import torch 
import torch
from torch.utils.data import DataLoader

from core.simulator import Simulator 
from algorithms.nrms_model import NRMS_Model
from utils.data_util import read_data, NewsDataset

class NRMS_Sim(Simulator): 
    def __init__(self, device, args, pretrained_mode=True, name='NRMS_Simulator'): 
        """
        Args:
            pretrained_mode: bool, True: load from a pretrained model, False: not pretrained model 
        """
        self.pretrained_mode = pretrained_mode 
        self.name = name 
        self.device = device 

        # preprocessed data 
        self.nid2index, _, self.news_index, embedding_matrix, _, _ = read_data(args) 

        # model 
        self.model = NRMS_Model(embedding_matrix).to(self.device)
        if self.pretrained_mode == 'pretrained':
            self.model.load_state_dict(torch.load(args.sim_path)) 

        # news_dataset 
        self.news_dataset = NewsDataset(self.news_index) 


    def get_news_vec(self, batch_size):
        news_dl = DataLoader(self.news_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        news_vecs = []
        for news in news_dl:
            news = news.to(self.device)
            news_vec = self.model.text_encoder(news).detach().cpu().numpy()
            news_vecs.append(news_vec)
        news_vecs = np.concatenate(news_vecs)

        return news_vecs

    # def get_user_vec(self, model, batch_sam, news_vecs, batch_nid2index):
    #     user_dataset = UserDataset(batch_sam, news_vecs, batch_nid2index)
    #     user_vecs = []
    #     user_dl = DataLoader(user_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    #     for his_tsp in user_dl:
    #         his, tsp = his_tsp
    #         his = his.to(device)
    #         user_vec = model.user_encoder(his).detach().cpu().numpy()
    #         user_vecs.append(user_vec)
    #         # print(tsp)
    #     user_vecs = np.concatenate(user_vecs)

    #     return user_vecs

