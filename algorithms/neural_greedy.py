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
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class SingleStageNeuralGreedy(ContextualBanditLearner):
    def __init__(self,device, args, name='SingleStageNeuralGreedy'):
        """Use NRMS model. 
        """
        # preprocessed data 
        self.nid2index, self.word2vec, self.nindex2vec = load_word2vec(args, 'utils')
        self.device = device 
        super(SingleStageNeuralGreedy, self).__init__(args, name) 
        self.name = name 
        
        self.n_inference = 1
        self.preinference_mode = self.args.preinference_mode

        topic_news = load_cb_topic_news(args) # dict, key: subvert; value: list nIDs 
        cb_news = []
        for k,v in topic_news.items():
            cb_news.append(l.strip('\n').split("\t")[0] for l in v) # get nIDs 
        cb_news = [item for sublist in cb_news for item in sublist]
        self.cb_news=cb_news
        
        # model 
        self.model = NRMS_Model(self.word2vec).to(self.device)
        self.model.eval()

        # if preinference_mode: # pre-generate news embeddings
        #     self.news_embs = self._get_news_embs()
        self.news_embs = []

    def run_eva(self):
        """Evaluate model on valid data
        """
        import os
        import sys
        module_path = os.path.abspath(os.path.join('..'))
        if module_path not in sys.path:
            sys.path.append(module_path)
        from thanh_preprocess import eva
        import pickle

        with open(os.path.join(self.args.root_data_dir, self.args.dataset, 'utils', "valid_contexts.pkl"), "rb") as f:
            valid_sam = pickle.load(f)
        val_scores = eva(self.args, self.model, valid_sam, self.nid2index, self.nindex2vec)
        val_auc, val_mrr, val_ndcg, val_ndcg10, ctr = [np.mean(i) for i in list(zip(*val_scores))]
        print(f"Debug: Evaluate model on valid data -- auc: {val_auc:.4f}, mrr: {val_mrr:.4f}, ndcg5: {val_ndcg:.4f}, ndcg10: {val_ndcg10:.4f}, ctr: {ctr:.4f}")
           

    def _get_news_embs(self):
        print('Inference news {} times...'.format(self.n_inference))
        news_dataset = NewsDataset(self.nindex2vec) 
        news_dl = DataLoader(news_dataset,batch_size=1024, shuffle=False, num_workers=2)
        
        self.news_embs = []
        for i in range(self.n_inference): 
            news_vecs = []
            for news in news_dl: # @TODO: avoid for loop
                news = news.to(self.device)
                news_vec = self.model.text_encoder(news).detach().cpu().numpy()
                news_vecs.append(news_vec)
            self.news_embs.append(np.concatenate(news_vecs))

            # print('Debug news embedding of # {} : {}'.format(i,np.concatenate(news_vecs)[0][:20]))
        # return np.concatenate(self.news_embs) # (n_inference, 130381, 400)

    # def _get_news_embs(self, news_vecs, user_samples): 
    #     """Transform user_samples into representation vectors. 

    #     Args:
    #         user_samples: a list of (poss, negs, his, uid, tsp) 

    #     Return: 
    #         user_vecs: [None, dim]
    #     """
    #     user_dataset = UserDataset(self.args, user_samples, news_vecs, self.nid2index)
    #     user_vecs = []
    #     user_dl = DataLoader(user_dataset, batch_size=min(1024, len(user_samples)), shuffle=False, num_workers=2)

    #     for his_tsp in user_dl:
    #         his, tsp = his_tsp
    #         his = his.to(self.device)
    #         user_vec = self.model.user_encoder(his).detach().cpu().numpy()
    #         user_vecs.append(user_vec)
    #         # print(tsp)
    #     return np.concatenate(user_vecs)

    def _get_user_embs(self, uid, i):
        # get user vect 
        
        h = self.clicked_history[uid]
        h = h + [0] * (self.args.max_his_len - len(h))
        h = self.news_embs[i][np.array(h)]
        h = torch.Tensor(h[None,:,:]).to(self.device)

        user_emb = self.model.user_encoder(h).detach().cpu().numpy()
        return user_emb

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
        # print('Debug ft_sam: ', ft_sam)
        if len(ft_sam) > 0:
            print('Updating the internal model of the bandit!')
            ft_ds = TrainDataset(self.args, ft_sam, self.nid2index, self.nindex2vec)
            ft_dl = DataLoader(ft_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
            
            # do one epoch only
            loss = 0
            self.model.train()
            ft_loader = tqdm(ft_dl)
            for cnt, batch_sample in enumerate(ft_loader):
                candidate_news_index, his_index, label = batch_sample
                candidate_news_index = candidate_news_index.to(self.device)
                his_index = his_index.to(self.device)
                label = label.to(self.device)
                bz_loss, y_hat = self.model(candidate_news_index, his_index, label)

                loss += bz_loss.detach().cpu().numpy()
                optimizer.zero_grad()
                bz_loss.backward()

                optimizer.step()  
            self._get_news_embs()
            self.data_buffer = [] # reset data buffer
        else:
            print('Skip update cb learner due to lack valid samples!')

    def update(self, topics, items, rewards, mode = 'item', uid = None):
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
        print('size(data_buffer): {}'.format(len(self.data_buffer)))
        if mode == 'item':
            self.train() 
            

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
            assert self.n_inference == 1
            user_vecs = self._get_user_embs(uid, 0) # (b,d)
            scores = self.news_embs[0][cand_news] @ user_vecs.T # (n,b) 
            nid_argmax = np.argsort(scores.squeeze(-1))[::-1][:m].tolist() # (len(uids),)
            rec_itms = [cand_news[n] for n in nid_argmax]
            return rec_itms 
        else:
            batch_size = min(self.args.max_batch_size, len(cand_news))
            # get user vect 
            h = self.clicked_history[uid]
            h = h + [0] * (self.args.max_his_len - len(h))
            h = self.nindex2vec[h]
            h = torch.Tensor(h[None,:,:])
            sed = SimEvalDataset2(self.args, cand_news, self.nindex2vec)
            rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=4) 

            scores = []
            for cn in rdl:
                score = self.model.forward(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, compute_loss=False)
                scores.append(score.detach().cpu().numpy()) 
            scores = np.concatenate(scores).squeeze(-1)
            # print(scores.shape)   

            nid_argmax = np.argsort(scores)[::-1][:m].tolist() # (len(uids),)
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
        #TODO: reset the internal model here for each instance of `ContextualBanditLearner`
        self.model = NRMS_Model(self.word2vec).to(self.device)