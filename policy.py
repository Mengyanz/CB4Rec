from pathlib import Path
import time
import datetime
from tqdm import tqdm
from collections import defaultdict, Counter
import copy
import random
import re
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import pickle
from datetime import datetime 
import math
import uncertainty_toolbox as utc

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from model import NRMS
from dataloader import TrainDataset, NewsDataset, UserDataset

class CB_sim():
    def __init__(
        self, model_path, simulator_path, out_path, device,
        news_index, nid2index, embedding_matrix,
        finetune_batch_size = 32, eva_batch_size = 1024, dropout_flag = True, 
        name = 'nrms'
    ):
        self.model = NRMS(embedding_matrix).to(self.device)
        self.model.load_state_dict(torch.load(model_path/f'{name}.pkl'))
        
        # TODO: change simulator to PLM
        self.simulator = NRMS(embedding_matrix).to(self.device)
        self.simulator.load_state_dict(torch.load(simulator_path/f'{name}.pkl'))

        self.news_index = news_index
        self.nid2index = nid2index

        self.out_path = out_path
        self.dropout_flag = dropout_flag
        
        self.finetune_batch_size = finetune_batch_size
        self.eva_batch_size = eva_batch_size
        self.date_format_str = '%m/%d/%Y %I:%M:%S %p'
        
        self.device = device

    
    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('dropout'):
                print(m)
                m.train() 
        
    def get_news_vec(self, model, batch_news_index, batch_size):
        batch_news_dataset = NewsDataset(batch_news_index)
        news_dl = DataLoader(batch_news_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        news_vecs = []
        for news in tqdm(news_dl):
            news = news.to(self.device)
            news_vec = model.text_encoder(news).detach().cpu().numpy()
            news_vecs.append(news_vec)
        news_vecs = np.concatenate(news_vecs)

        return news_vecs

    def get_user_vec(self, model, batch_sam, news_vecs, batch_nid2index):
        user_dataset = UserDataset(batch_sam, news_vecs, batch_nid2index)
        user_vecs = []
        user_dl = DataLoader(user_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for his_tsp in tqdm(user_dl):
            his, tsp = his_tsp
            his = his.to(self.device)
            user_vec = model.user_encoder(his).detach().cpu().numpy()
            user_vecs.append(user_vec)
            # print(tsp)
        user_vecs = np.concatenate(user_vecs)

        return user_vecs
        
    def get_cand_news(self, t, poss, negs, m = 10):
        """
        Generate candidates news based on CTR.

        t: string, impr time
        poss: list, positive samples in impr
        negs: list, neg samples in impr
        m: int, number of candidate news to return

        Return: array, candidate news id 
        """
        t = datetime.strptime(t,date_format_str)
        tidx = int((t - start_time).total_seconds()/interval_time)

        nonzero = news_ctr[:,tidx -1][news_ctr[:, tidx-1] > 0]
        nonzero_idx = np.where(news_ctr[:, tidx-1] > 0)[0]
        # print(nonzero_idx)

        nonzero = nonzero/nonzero.sum()
        assert (nonzero.sum() - 1) < 1e-3
        # print(np.sort(nonzero)[-5:])

        # sampling according to normalised ctr in last one hour
        sample_nids = np.random.choice(nonzero_idx, size = m, replace = False, p = nonzero)
        # REVIEW: check whether the sampled nids are reasonable
        
        # print(news_ctr[sample_nidx, tidx-1])
        # plt.hist(nonzero)
        # print(poss)
        # print(negs)
        poss_nids = np.array([self.nid2index[n] for n in poss])
        negs_nids = np.array([self.nid2index[n] for n in negs])
        # print('get cand news')
        # print(sample_nids)
        # print(poss_nids)
        # print(negs_nids)

        return np.concatenate([sample_nids, poss_nids, negs_nids])
        
#     t =  sorted_train_sam[1500][-1]
#     gene_news_pool(t, 20)

    def construct_trainable_samples(self, samples):
        tr_samples = []
        for l in samples:
            pos_imp, neg_imp, his, uid, tsp = l    
            for pos in list(pos_imp):
                tr_samples.append([pos, neg_imp, his, uid, tsp])
        return tr_samples

    def finetune(self, ft_sam):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        ft_sam = self.construct_trainable_samples(ft_sam)
        ft_ds = TrainDataset(ft_sam, self.nid2index, self.news_index)
        ft_dl = DataLoader(ft_ds, batch_size=self.finetune_batch_size, shuffle=True, num_workers=0)
        for ep in range(epoch):
            loss = 0
            accuary = 0.0
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

                if cnt % 10 == 0:
                    ft_loader.set_description(f"[{cnt}]steps loss: {loss / (cnt+1):.4f} ")
                    ft_loader.refresh() 

    def get_ucb_score(self, exper_id, y_score, t):
        ucb = []
        beta_t = 1

        mean = np.asarray(y_score).mean(axis = 0)
        std = np.asarray(y_score).std(axis = 0)
        ucb_score = mean + beta_t * std

        rec_nids = np.argsort(ucb_score)[-k:]
        return rec_nids

    def epsilon_greedy(self, exper_id, y_score, k):
        rec = []
        y_score = y_score[0]
        p = np.random.rand(k)
        n_greedy = int(len(p[p > self.policy_para]))
        greedy_nids = np.argsort(y_score)[-n_greedy:]
        eps_nids = np.random.choice(np.array(list(set(range(len(y_score))) - set(greedy_nids))), size = k - n_greedy, replace=False)
        rec_nids= np.concatenate([greedy_nids, eps_nids])
        return rec_nids
        
    def sigmoid(self, x):
        return np.array([1/(1 + math.exp(-i)) for i in x])
                
    def rec(self, batch_sam, batch_id, exper_id, k = 8):
        """
        Simulate recommendations
        """
        if self.policy == 'epsilon_greedy':
            self.n_inference = 1
        
        self.model.eval()
        if self.dropout_flag:
            self.enable_dropout()

        y_scores = defaultdict(list) # key: sam_id, value: list of n_inference scores 
        cand_nids_dict = {} # key: sam_id, value: array of candidate news ids
        start_id = self.eva_batch_size * batch_id

        with torch.no_grad():
            sim_news_vecs = self.get_news_vec(self.simulator, self.news_index, batch_size=self.eva_batch_size)
            sim_user_vecs = self.get_user_vec(self.simulator, batch_sam, sim_news_vecs, self.nid2index) 

            # generate scores with uncertainty (dropout during inference)
            for _ in tqdm(range(self.n_inference)):
                # TODO: speed up - only pass batch news index
                news_vecs = self.get_news_vec(self.model, self.news_index, batch_size=self.eva_batch_size)
                user_vecs = self.get_user_vec(self.model, batch_sam, news_vecs, self.nid2index)
                
                for i in range(len(batch_sam)):
                    t = start_id + i
                    poss, negs, his, uid, tsq = batch_sam[i]     
                    user_vec = user_vecs[i]
                    
                    if t not in cand_nids_dict.keys():
                        # cand set (is a randomised set) keeps same for inference times
                        cand_nids = self.get_cand_news(tsq, poss, negs)
                        cand_nids_dict[t] = cand_nids
                        # print(cand_nids)
                        news_vec = news_vecs[cand_nids]
                        sim_user_vec = sim_user_vecs[i]
                        sim_news_vec = sim_news_vecs[cand_nids]
                        sim_y_score = np.sum(np.multiply(sim_news_vec, sim_user_vec), axis=1)
                        # assume the user would at most click half of the recommended news
                        opt_nids = np.argsort(sim_y_score)[-int(k/2):] 
                        # print(opt_nids)
                        # print(self.sigmoid(sim_y_score[opt_nids]))
                        opt_nids = opt_nids[self.sigmoid(sim_y_score[opt_nids]) > 0.5]
                        # print(opt_nids)
                        self.opts[exper_id].append(opt_nids)   
                    
                    news_vec = news_vecs[cand_nids_dict[t]]
                    y_score = np.sum(np.multiply(news_vec, user_vec), axis=1)
                    y_scores[t].append(y_score)

        # generate recommendations and calculate rewards
        for i in range(len(batch_sam)):
            t = start_id + i
            _, _, his, uid, tsq = batch_sam[i]  

            if self.policy == 'ucb':
                rec_nids = self.get_ucb_score(exper_id, y_scores[t], t)
            elif self.policy == 'epsilon_greedy':
                rec_nids = self.epsilon_greedy(exper_id, y_scores[t], k)
            else:
                raise NotImplementedError

            self.recs[exper_id].append(rec_nids) 
            
            opt_nids = self.opts[exper_id][i]
 
            assert len(set(rec_nids)) == k
            # assert len(set(opt_nids)) == k

            reward = len(set(rec_nids) & set(opt_nids)) # reward as the overlap between rec and opt set
            # self.cumu_reward += reward
            self.rewards[exper_id, t] = reward
            self.opt_rewards[exper_id, t] = len(set(opt_nids))

            rec_poss = [rec_nid for rec_nid in rec_nids if rec_nid in opt_nids]
            # let all 
            rec_negs = list(set(cand_nids_dict[t]) - set(rec_poss))
            
            self.rec_sam.append([rec_poss, rec_negs, his, uid, tsq])
        
    def run_exper(self, test_sam, num_exper = 10, n_inference = 2, policy = 'ucb', policy_para = 0.1):
        
        num_sam = len(test_sam)
        n_batch = math.ceil(float(num_sam)/self.eva_batch_size)
        self.rec_sam = []
        self.rewards = np.zeros((num_exper, num_sam))
        self.opt_rewards = np.zeros((num_exper, num_sam))
        self.recs = defaultdict(list)
        self.opts = defaultdict(list)

        self.n_inference = n_inference
        self.policy = policy
        self.policy_para = policy_para

        update_time = None
        update_batch = 0
        

        for j in range(num_exper):
            # self.cumu_reward = 0
            
            for i in range(n_batch):
                upper_range = min(self.eva_batch_size * (i+1), len(test_sam))
                batch_sam = test_sam[self.eva_batch_size * i: upper_range]

                self.rec(batch_sam, i, j)

                if update_time is None:
                    update_time = datetime.strptime(str(batch_sam[0][-1]), self.date_format_str)
                    print('init time: ', update_time)

                batch_time = datetime.strptime(str(batch_sam[-1][-1]), self.date_format_str)
                if (batch_time- update_time).total_seconds() > 3600:
                    lower_range = self.eva_batch_size * update_batch
                    ft_sam = self.rec_sam[lower_range: upper_range]
                    if upper_range - lower_range > 512:
                        print('finetune with: '  + str(lower_range) + ' ~ ' + str(upper_range))
                        self.finetune(ft_sam=ft_sam)

                        update_time = batch_time
                        update_batch = i + 1
                        print('update at: ', update_time)
                    else: 
                        print('no finetune due to insufficient samples: ', str(upper_range - lower_range))

        self.save_results()

    def save_results(self):
        policy_name = self.policy + '_' + str(self.policy_para)
        torch.save(self.model.state_dict(), os.path.join(model_path, (policy_name + f'_{name}_finetune.pkl')))
        with open(os.path.join(self.out_path, (policy_name + "_recs.pkl")), "wb") as f:
            pickle.dump(self.recs, f)
        with open(os.path.join(self.out_path, (policy_name + "_opts.pkl")), "wb") as f:
            pickle.dump(self.opts, f)
        with open(os.path.join(self.out_path, (policy_name + "_rewards.pkl")), "wb") as f:
            pickle.dump(self.rewards, f)
        with open(os.path.join(self.out_path, (policy_name+ "_opt_rewards.pkl")), "wb") as f:
            pickle.dump(self.opt_rewards, f)