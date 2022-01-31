"""Define NRMS simulator. """

import math 
import numpy as np 
import torch 
from torch.utils.data import DataLoader

from core.simulator import Simulator 
from algorithms.nrms_model import NRMS_Model
from utils.data_util import read_data, NewsDataset, UserDataset, load_word2vec, SimEvalDataset, SimEvalDataset2


class NRMS_Sim(Simulator): 
    def __init__(self, device, args, pretrained_mode=True, name='NRMS_Simulator'): 
        """
        Args:
            pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 
        """
        super(NRMS_Sim, self).__init__(name)

        self.pretrained_mode = pretrained_mode 
        self.name = name 
        self.device = device 
        self.args = args 

        # preprocessed data 
        # self.nid2index, _, self.news_index, embedding_matrix, self.train_samples, self.valid_samples = read_data(args) 

        self.nid2index, word2vec, self.nindex2vec = load_word2vec(args)

        # model 
        self.model = NRMS_Model(word2vec).to(self.device)
        if self.pretrained_mode == 'pretrained':
            self.model.load_state_dict(torch.load(args.sim_path)) 


    def reward(self, uid, news_indexes): 
        """Returns a simulated reward. 

        Args:
            uid: str, user id 
            news_indexes: a list of item index (not nID, but its integer version)

        Return: 
            rewards: (n,m) of 0 or 1 
        """
        batch_size = min(self.args.max_batch_size, len(news_indexes))

        h = self.clicked_history[uid]
        h = h + [0] * (self.args.max_his_len - len(h))
        h = self.nindex2vec[h]

        h = torch.Tensor(h[None,:,:])

        sed = SimEvalDataset2(self.args, news_indexes, self.nindex2vec)
        rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=4) 

        scores = []
        for cn in rdl:
            score = self.model(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, compute_loss=False)
            scores.append(score.detach().cpu().numpy()) 
        scores = np.concatenate(scores)  
        p = sigmoid(scores) 
        rewards = np.random.binomial(size=p.shape, n=1, p=p)
        return rewards 

def sigmoid(u):
    return 1/(1+np.exp(-u))