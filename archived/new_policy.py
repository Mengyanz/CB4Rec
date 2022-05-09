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
        self.num_round = args.num_round
        self.num_inference = args.n_inference 
        self.policy = args.policy
        self.policy_para = args.policy_para 
        self.m = args.m # number of recommendations

        self.cb_uids = np.load(args.cb_users)
        self.cb_news = np.load(args.cb_news)
        self.cb_topics = np.load(args.cb_topics)
        self.init_paras()

    def init_paras(self):
        self.alphas = {}
        self.betas = {}

        for topic in self.cb_topics:
            self.alphas[topic] = 1
            self.betas[topic] = 1

    def load_simulator(self, args):
        if self.sim_type == 'nrms':
            self.simulator = NRMS(embedding_matrix).to(self.device)
            self.simulator.load_state_dict(torch.load(args.sim_path))
        elif self.sim_type == 'ips':
            pass
        elif self.sim_type == 'unium':      
            pass
            # TODO: Thanh to implement 
            

    def enable_dropout(self):
        # TODO: check whether it truly controls dropout
        for m in self.model.modules():
            if m.__class__.__name__.startswith('dropout'):
                print(m)
                m.train() 
        
    def get_news_vec(self, model, batch_news_index):
        batch_news_dataset = NewsDataset(batch_news_index)
        news_dl = DataLoader(batch_news_dataset, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers)
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
        user_dl = DataLoader(user_dataset, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers)

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
        ft_dl = DataLoader(ft_ds, batch_size=self.finetune_batch_size, shuffle=True, num_workers=self.args.num_workers)
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

        if self.m == 'all':
            k = len(mean)
        else:
            k = self.m

        recs = np.argsort(ucb_score)[-k:][::-1]
        
        # return local idxs
        return recs

    def epsilon_greedy(self, exper_id, batch_id, y_score):
        rec = []
        y_score = y_score[0]
        # DEBUG: 
        if self.m == 'all':
            k = len(y_score)
        else:
            k = self.m
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

        
    def item_rec(self, rec_topic):
        """
        input: rec_topic, outpout: rec_item
        """
        if self.policy == 'epsilon_greedy':
            # current epsilon greedy only relies on uniformly random sampling, so no need to inference multiple times
            self.num_inference = 1
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
            for _ in tqdm(range(self.num_inference)):
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
            # print('self.m: ', self.m)
            # assert len(set(rec_nids)) == self.m
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

    def topic_rec(self, u):
        """Input: uid; output: recommended one topic
        """
        ss =[] 
        for topic in self.active_topics:
            s = np.random.beta(a= self.alpha[topic], b= self.beta[topic])
            ss.append(s)
        rec_topic = self.active_topics[np.argmax(ss)]
        return rec_topic

    def get_simulated_rewards(self, rec_set):
        """
        Input: 
            rec_set: list
                list of recommended news ids
        Outpout: 
            rewards: list
                list of rewards of each recommended news
        """
        # TODO: Thanh to implement 
        
    def run_exper(self):
        """CB simulation to cb_uids (for num_exper simulations; num_round passes over users;)
        """
        for j in range(self.num_exper):
            for i in range(self.num_round):
                for u in self.cb_uids:
                    rec_topics = []
                    rec_items = []
                    self.active_topics = self.cb_topics
                    while len(rec_items) < self.m:
                        rec_topic = self.topic_rec(u)
                        rec_topics.append(rec_topic)
                        self.active_topics.remove(rec_topic)

                        rec_item = self.item_rec(rec_topic)
                        rec_items.append(rec_item)

                    rewards = self.get_simulated_reward(rec_items)

                    for i, topic in enumerate(rec_topics):
                        assert rewards[i] in {0,1}
                        self.alphas[topic] += rewards[i]
                        self.betas[topic] += 1 - rewards[i]                        
                # TODO: update models
