"""Define a NRMS based greedy recommendation policy. """

import math 
import numpy as np 
from collections import defaultdict
import torch 
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.nrms_model import NRMS_Model
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news, SimEvalDataset

class SingleStageNeuralGreedy(ContextualBanditLearner):
    def __init__(self,device, args, rec_batch_size = 1, pretrained_mode=True, name='SingleStageNeuralGreedy'):
        """Use NRMS model. 
            Args:
                rec_batch_size: int, recommendation size. 
                n_inference: int, number of Monte Carlo samples of prediction. 
                pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 

        """
        super(SingleStageNeuralGreedy, self).__init__(args, rec_batch_size, name) 
        self.pretrained_mode = pretrained_mode 
        self.name = name 
        self.device = device 

        # preprocessed data 
        self.nid2index, word2vec, self.nindex2vec = load_word2vec(args)
        topic_news = load_cb_topic_news(args) # dict, key: subvert; value: list nIDs 
        self.candidate_news = []
        for k,v in topic_news.items():
            self.candidate_news.append(l.strip('\n').split("\t")[0] for l in v) # get nIDs 
        self.candidate_news = [item for sublist in self.candidate_news for item in sublist]

        # model 
        self.model = NRMS_Model(word2vec).to(self.device)
        if self.pretrained_mode == 'pretrained':
            self.model.load_state_dict(torch.load(args.learner_path)) 

    def construct_trainable_samples(self):
        """construct trainable samples which will be used in NRMS model training
        from self.h_contexts, self.h_actions, self.h_rewards
        """
        tr_samples = []

        for i, l in enumerate(self.data_buffer):
            poss, negs, his, uid, tsp = l
            if len(poss) > 0 and len(negs) > 0:  # TODO: change when use BCE
                for pos in poss:
                    tr_samples.append([pos, negs, his, uid, tsp])
        return tr_samples

    def train(self):
        
        # update learner
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        ft_sam = self.construct_trainable_samples()
        if len(ft_sam) > 0:
            print('Updating the internal model of the bandit!')
            ft_ds = SimTrainDataset(self.args, ft_sam, self.nid2index, self.nindex2vec)
            ft_dl = DataLoader(ft_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
            
            # do one epoch only
            loss = 0
            self.model.train()
            ft_loader = tqdm(ft_dl)
            for cnt, batch_sample in enumerate(ft_loader):
                candidate_news_index, his_index, label = batch_sample
                sample_num = candidate_news_index.shape[0]
                candidate_news_index = candidate_news_index.to(self.device)
                his_index = his_index.to(self.device)
                label = label.to(self.device)
                bz_loss, y_hat = self.model(candidate_news_index, his_index, label)

                loss += bz_loss.detach().cpu().numpy()
                optimizer.zero_grad()
                bz_loss.backward()

                optimizer.step()  
        else:
            print('Skip update cb learner due to lack valid samples!')

    def update(self, topics, items, rewards, mode = 'item'):
        """Update its internal model. 

        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`

        @TODO: they recommend `rec_batch_size` topics 
            and each of the topics they recommend an item (`rec_batch_size` items in total). 
            What if one item appears more than once in the list of `rec_batch_size` items? 
        """
        self.train() 

    def item_rec(self, uids, cand_news, m = 1): 
        """
        Args:
            uids: a list of str uIDs 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        batch_size = min(16, len(uids))
        candidate_news = self.nindex2vec[[n for n in cand_news]] 
        candidate_news = torch.Tensor(candidate_news[None,:,:]).repeat(batch_size,1,1)
        sed = SimEvalDataset(self.args, uids, self.nindex2vec, self.clicked_history)
        #TODO: Use Dataset is clean and good when len(uids) is large. When len(uids) is small, is it faster to not use Dataset?
        rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=4) 

        
        scores = []
        for h in rdl:
            # TODO: out of memory when use all news as candidate news
            score = self.model.forward(candidate_news.to(self.device), h.to(self.device), None, compute_loss=False)
            scores.append(score.detach().cpu().numpy()) 
        scores = np.concatenate(scores) # (len(uids), len(cand_news))

        nid_argmax = np.argsort(scores, axis=1)[::-1][:m].tolist() # (len(uids),)
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

        # print(self.nid2index)
        # cand_news = []
        # for n in self.candidate_news:
        #     cand_news.append(self.nid2index[n])
        cand_news = [self.nid2index[n] for n in self.candidate_news]
        rec_items = self.item_rec(uids, cand_news, self.rec_batch_size)

        
        return np.empty(0), rec_items
    