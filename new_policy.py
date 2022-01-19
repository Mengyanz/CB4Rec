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
# import uncertainty_toolbox as utc

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from model import NRMS
from dataloader import TrainDataset, NewsDataset, UserDataset
# from run import read_data
import logging


date_format_str = '%m/%d/%Y %I:%M:%S %p'


class CB_sim():
    def __init__(
        self, args, device, name = 'nrms'
    ):

        self.nid2index, self.news_index, embedding_matrix, sorted_train_sam, self.sorted_valid_sam = read_data(args)
        self.device = device
        self.model = NRMS(embedding_matrix).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(args.model_path, args.dataset, f'{name}.pkl')))
        self.sim_type = args.sim_type
        self.load_simulator(args)

        self.date_format_str = args.data_format_str
        self.start_time = datetime.strptime(sorted_train_sam[0][-1],self.date_format_str)
        self.interval_time = args.interval_time 

        self.finetune_batch_size = args.batch_size
        self.eva_batch_size = args.eva_batch_size
        self.lr = args.lr
        self.epoch = args.epochs
        self.out_path = args.out_path
        self.dropout_flag = args.dropout_flag
        self.finetune_flag = args.finetune_flag

        self.num_exper = args.num_exper
        self.n_inference = args.n_inference 
        self.policy = args.policy
        self.policy_para = args.policy_para 
        self.k = args.k

    def load_simulator(self, args):
        if self.sim_type == 'nrms':
            self.simulator = NRMS(embedding_matrix).to(self.device)
            self.simulator.load_state_dict(torch.load(args.sim_path))
        elif self.sim_type == 'ips':
            pass
        elif self.sim_type == 'unium':      
            pass
            

    def enable_dropout(self):
        # TODO: check whether it truly controls dropout
        for m in self.model.modules():
            if m.__class__.__name__.startswith('dropout'):
                print(m)
                m.train() 
        
    def get_news_vec(self, model, batch_news_index):
        batch_news_dataset = NewsDataset(batch_news_index)
        news_dl = DataLoader(batch_news_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        news_vecs = []
        for news in news_dl:
            news = news.to(device)
            news_vec = model.text_encoder(news).detach().cpu().numpy()
            news_vecs.append(news_vec)
        news_vecs = np.concatenate(news_vecs)

        return news_vecs

    def get_user_vec(self, model, batch_sam, news_vecs, batch_nid2index):
        user_dataset = UserDataset(batch_sam, news_vecs, batch_nid2index)
        user_vecs = []
        user_dl = DataLoader(user_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for his_tsp in user_dl:
            his, tsp = his_tsp
            his = his.to(device)
            user_vec = model.user_encoder(his).detach().cpu().numpy()
            user_vecs.append(user_vec)
            # print(tsp)
        user_vecs = np.concatenate(user_vecs)

        return user_vecs
        
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
                candidate_news_index = candidate_news_index.to(device)
                his_index = his_index.to(device)
                label = label.to(device)
                bz_loss, y_hat = self.model(candidate_news_index, his_index, label)

                loss += bz_loss.detach().cpu().numpy()
                optimizer.zero_grad()
                bz_loss.backward()

                optimizer.step()

                if cnt % 10 == 0:
                    ft_loader.set_description(f"[{cnt}]steps loss: {loss / (cnt+1):.4f} ")
                    ft_loader.refresh() 

    def get_ucb_score(self, exper_id, batch_id, y_score):
        mean = np.asarray(y_score).mean(axis = 0)
        std = np.asarray(y_score).std(axis = 0)
        # print('mean: ', mean)
        # print('std: ', std)
        if self.policy_para == 'logt':
            # REVIEW: whether to use batch id?
            para = np.log(batch_id + 2) * 0.1
            print('policy para: ', para)
        else:
            para = self.policy_para
        ucb_score = mean + para * std

        if self.k == 'all':
            k = len(mean)
        else:
            k = self.k

        recs = np.argsort(ucb_score)[-k:][::-1]
        
        # return local idxs
        return recs

    def epsilon_greedy(self, exper_id, batch_id, y_score):
        rec = []
        y_score = y_score[0]
        # DEBUG: 
        if self.k == 'all':
            k = len(y_score)
        else:
            k = self.k
        p = np.random.rand(k)
        if self.policy_para == '1overt':
            # REVIEW: whether to use batch id?
            para = 1.0/(2 *(batch_id + 1))
            print('policy para: ', para)
        else:
            para = self.policy_para
        n_greedy = int(len(p[p > para]))
        if n_greedy == 0:
            greedy_nids = []
        else:
            greedy_nids = np.argsort(y_score)[-n_greedy:][::-1]

        print('greedy nids: ', greedy_nids)
        if n_greedy < k:
            if len(list(set(list(range(len(y_score)))) - set(greedy_nids))) == 0:
                print('y_score: ', y_score)
                print('n_greedy: ', n_greedy)
                print(np.array(list(set(list(range(len(y_score)))) - set(greedy_nids))))

            eps_nids = np.random.choice(np.array(list(set(list(range(len(y_score)))) - set(greedy_nids))), size = k - n_greedy, replace=False)
            print('eps nids: ', eps_nids)
            rec_nids= np.array([int(i) for i in np.concatenate([greedy_nids, eps_nids])])
            return rec_nids
        else:
            return greedy_nids 
        
    def sigmoid(self, x):
        return np.array([1/(1 + math.exp(-i)) for i in x])

    def get_labels(self, sam, cand):
        """
        prepare candidate news and get labels
        """
        sim_labels = []
        labels = []

        # if self.sim_flag: # if use simulated labels, then return simulated y scores
        sim_news_vecs = self.get_news_vec(self.simulator, self.news_index)
        sim_user_vecs = self.get_user_vec(self.simulator, sam, sim_news_vecs, self.nid2index) 

        for i in range(len(sam)):
            poss, negs, _, uid, tsq = sam[i] 
            
            sim_user_vec = sim_user_vecs[i]
            sim_news_vec = sim_news_vecs[cand[i]]
            sim_y_score = np.sum(np.multiply(sim_news_vec, sim_user_vec), axis=1)
            # REVIEW: how to choose opt set
            # assume the user would at most click half of the recommended news
            # opt_nids = np.argsort(sim_y_score)[-max(int(self.k/2), 1):]
            # opt_nids = np.argsort(sim_y_score)[-max(int(len(sim_y_score)/2), 1):]  
            # opt_nids = np.argsort(sim_y_score)
            # opt_nids = opt_nids[self.sigmoid(sim_y_score[opt_nids]) > 0.5]
            # print(opt_nids)

            sim_labels.append(sim_y_score) 
        # else: # otherwise, use impression labels
        for i in range(len(sam)):
            poss, negs, _, uid, tsq = sam[i]    
            label = [1] * len(poss) + [0] * len(negs)
            labels.append(np.array(label))
        return sim_labels, labels

    def get_batch_labels(self, batch_sam, batch_recs):
        """Predict from batch_sam and batch_recs by simulator
        """
        labels = []
        sim_news_vecs = self.get_news_vec(self.simulator, self.news_index)
        sim_user_vecs = self.get_user_vec(self.simulator, batch_sam, sim_news_vecs, self.nid2index) 

        for i in range(len(batch_sam)):
            poss, negs, _, _, tsq = batch_sam[i] 
            
            sim_user_vec = self.sim_user_vecs[i]
            sim_news_vec = self.sim_news_vecs[batch_recs[i]]
            sim_y_score = np.sum(np.multiply(sim_news_vec, sim_user_vec), axis=1)
            # assume the user would at most click half of the recommended news
            # reward = 1 if self.sigmoid(sim_y_score) > 0.5 else 0
            # opt_nids = opt_nids[self.sigmoid(sim_y_score[opt_nids]) > 0.5]
            # print(opt_nids)

            labels.append(sim_y_score)
        return labels
        
    def rec(self, batch_sam, batch_cand, batch_id, exper_id):
        """
        Simulate recommendations
        """
        if self.policy == 'epsilon_greedy':
            # current epsilon greedy only relies on uniformly random sampling, so no need to inference multiple times
            self.n_inference = 1
            self.dropout_flag = False
        
        self.model.eval()
        if self.dropout_flag:
            self.enable_dropout()

        y_scores = defaultdict(list) # key: sam_id, value: list of n_inference scores 
        start_id = self.eva_batch_size * batch_id

        with torch.no_grad():
            
            """
            notes for speed up
            inf_idx, behav_idx, user_idx, news_idx
            1, 1, 1, 1
            1, 1, 1, 2
            1, 1, 1, 3
            1, 2, 1, 4

            n_inf, n1, user_dim
            n_inf, n1, news_dim

            n_inf, n1 = 
            """
            # generate labels from simulators
            # y_trues = self.get_reward(batch_sam, batch_cand)

            # generate scores with uncertainty (dropout during inference)
            for _ in tqdm(range(self.n_inference)):
                # TODO: speed up - only pass batch news index
                news_vecs = self.get_news_vec(self.model, self.news_index) # batch_size * news_dim (400)
                user_vecs = self.get_user_vec(self.model, batch_sam, news_vecs, self.nid2index)
                
                for i in range(len(batch_sam)):
                    t = start_id + i # global idx 
                    poss, negs, _, uid, tsq = batch_sam[i]     
                    user_vec = user_vecs[i]
                    
                    news_vec = news_vecs[batch_cand[i]]
                    y_score = np.sum(np.multiply(news_vec, user_vec), axis=1)
                    y_scores[t].append(y_score)

        # generate recommendations and calculate rewards
        for i in range(len(batch_sam)):
            t = start_id + i
            _, _, his, uid, tsq = batch_sam[i]  

            if self.policy == 'ucb':
                recs = self.get_ucb_score(exper_id, batch_id, y_scores[t])
                print(recs)
                rec_nids = batch_cand[i][recs]
            elif self.policy == 'epsilon_greedy':
                recs = self.epsilon_greedy(exper_id, batch_id, y_scores[t])
                print(recs)
                rec_nids = batch_cand[i][recs]
            else:
                raise NotImplementedError

            # print('len(set(rec_nids)):', len(set(rec_nids)))
            # print('self.k: ', self.k)
            # assert len(set(rec_nids)) == self.k
            self.recs[exper_id].append(rec_nids) 
            
            # reward as the overlap between rec and opt set
            # reward = len(set(rec_nids) & set(opt_nids)) 
            # reward as auc score
            
            
            y_score = np.asarray(y_scores[t]).mean(axis = 0)
            sim_labels, labels = self.sim_labels[t][recs], self.labels[t][recs]
            # print(self.labels[t][recs])
            # print(y_score[recs])
            if all(i == 1 for i in labels):
                reward = 1
            elif all(i == 0 for i in labels):
                reward = 0
            else:
                reward = roc_auc_score(labels,y_score[recs])
            # print(reward)
            # self.cumu_reward += reward
            self.rewards[exper_id, t] = reward
            # TODO: next line only for single arm rec
            # self.opt_rewards[exper_id, t] = 1 if len(set(opt_nids)) > 0 else 0
            # DEBUG: 
            self.opt_rewards[exper_id, t] = roc_auc_score(labels,sim_labels)

            # REVIEW: put opt nids in rec poss as well?
            # rec_poss = [rec_nid for rec_nid in rec_nids if rec_nid in opt_nids]
            # rec_poss = [rec_nid for rec_nid in set(rec_nids).union(opt_nids)]
            # print('rec poss: ', rec_poss) 

            rec_poss = [rec_nids[np.argmax(y_score)]]
            rec_negs = list(set(rec_nids) - set(rec_poss))
            
            self.rec_sam.append([rec_poss, rec_negs, his, uid, tsq])
        
    def run_exper(self, cand_nidss):
        
        num_sam = len(self.sorted_val_sam)
        n_batch = math.ceil(float(num_sam)/self.eva_batch_size)
        self.rec_sam = []
        self.rewards = np.zeros((self.num_exper, num_sam))
        self.opt_rewards = np.zeros((self.num_exper, num_sam))
        self.recs = defaultdict(list)
        self.sim_labels, self.labels = self.get_labels(self.sorted_val_sam, cand_nidss)

        update_time = None
        update_batch = 0
      
        # opts = self.get_opt_set(self.sorted_val_sam, cand_nidss)
        

        for j in range(self.num_exper):
            # self.cumu_reward = 0
            print('exper: ', j)
            
            for i in range(n_batch):
                upper_range = min(self.eva_batch_size * (i+1), len(self.sorted_val_sam))
                batch_sam = self.sorted_val_sam[self.eva_batch_size * i: upper_range]
                batch_cand = cand_nidss[self.eva_batch_size * i: upper_range]

                self.rec(batch_sam, batch_cand, i, j)

                if self.finetune_flag:
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
            update_time = None
            update_batch = 0

        self.save_results()

    def save_results(self):
        folder_name = 'rec' + str(self.k) + '_ft' + str(self.finetune_flag)[0] + '_sim' + str(self.sim_type)[0]
        save_path = os.path.join(self.out_path, folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        policy_name = self.policy + '_' + str(self.policy_para)
        torch.save(self.model.state_dict(), os.path.join(save_path, (policy_name + f'_{name}_finetune.pkl')))
        # with open(os.path.join(self.out_path, (policy_name + "_recs.pkl")), "wb") as f:
        #     pickle.dump(self.recs, f)
        # with open(os.path.join(self.out_path, (policy_name + "_opts.pkl")), "wb") as f:
        #     pickle.dump(self.opts, f)
        with open(os.path.join(save_path, (policy_name + "_rewards.pkl")), "wb") as f:
            pickle.dump(self.rewards, f)
        with open(os.path.join(save_path, (policy_name+ "_opt_rewards.pkl")), "wb") as f:
            pickle.dump(self.opt_rewards, f)