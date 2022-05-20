"""Define a neural linear. """

import math 
import numpy as np 
import pandas as pd
from collections import defaultdict
import torch 
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import invgamma
import datetime
import pickle

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.neural_greedy import NeuralGreedy, Two_NeuralGreedy
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from algorithms.lr_model import LogisticRegression, LogisticRegressionAddtive, LogisticRegressionBilinear
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, GLMTrainDataset, load_word2vec, load_cb_topic_news,load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset
from torch.utils.tensorboard import SummaryWriter
import os
# from scipy.optimize import newton, root 

class Two_NeuralGLMAddUCB(Two_NeuralGreedy):
    def __init__(self, args, device, name='2_neuralglmadducb'):
        """Use NRMS model. 
        """
        super(Two_NeuralGLMAddUCB, self).__init__(args, device, name) 
        self.gamma = self.args.gamma
        self.dim = self.args.news_dim # topic and news has the same dim

        # item
        self.A =  np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))

        self.lr_model = LogisticRegressionAddtive(self.dim, 1).to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.data_buffer_lr = [] # for logistic regression

        # topic lr
        self.A_topic =  np.identity(n=self.dim)
        self.Ainv_topic = np.linalg.inv(self.A)
        self.b_topic = np.zeros((self.dim))

        self.lr_model_topic = LogisticRegressionAddtive(self.dim, 1).to(self.device)

        self.criterion = torch.nn.BCELoss()
        
        # debug 
        out_folder = os.path.join(args.result_path, 'runs') # store final results
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            
        out_path = os.path.join(out_folder, args.algo_prefix)
        self.writer = SummaryWriter(out_path) # https://pytorch.org/docs/stable/tensorboard.html
        self.round = 0

        self.topic_embs = []
        
    def pretrain_glm_learner(self, trial=None, mode = 'item', model_name = 'additive'):
        # NOTE: This intened to pre-train linear models with cb train data. However, we cannot do that since we remove the cb simulated users from the cb train data, and the linear models are disjoint, i.e. theta_u for per user. We can do this for our proposed shared model though.
        """pretrain the generalised linear models (if any) with the pretrained data, to initialise glm parameters.
        """
        out_model_path = os.path.join(self.args.root_proj_dir,  'cb_{}_pretrained_glm_models'.format(mode), model_name)
        if not os.path.exists(out_model_path):
            os.makedirs(out_model_path)
            
        load_path = os.path.join(out_model_path, 'indices_{}.pkl'.format(trial))
        if os.path.exists(load_path):
            if mode == 'topic':
                print('Load pre-trained CB GLM topic learner on this trial from ', load_path)
                self.topic_model.load_state_dict(torch.load(load_path))
            else:
                print('Load pre-trained CB GLM item learner on this trial from ', load_path)
                self.model.load_state_dict(torch.load(load_path))
        else:
            out_path = os.path.join(self.args.root_data_dir, self.args.dataset, 'utils')
        
            cb_train_fname = os.path.join(out_path, "cb_train_contexts_nuser=1000_splitratio=0.2_trial={}.pkl".format(trial))
            print('pretrain glm {} learner using {}'.format(mode, cb_train_fname))
            with open(cb_train_fname, "rb") as f:
                cb_train_sam = pickle.load(f)
            
            cb_train_uidfile = os.path.join(out_path, "cb_train_uid2index.pkl")
            # if os.path.exists(cb_train_uidfile):
            #     with open(os.path.join(out_path, "cb_train_uid2index.pkl"), "rb") as f:
            #         self.uid2index = pickle.load(f)
            # else:
            self.uid2index = {}
            index = 1
            for sample in cb_train_sam:
                _, _, _, uid, _ = sample
                if uid not in self.uid2index:
                    self.uid2index[uid] = index
                    index += 1
                # with open(cb_train_uidfile, "wb") as f:
                #     pickle.dump(self.uid2index, f)
            self.uindex2id = {v:k for k,v in self.uid2index.items()}
                
            self.train_glm(cb_train_sam, mode = mode, save_path=load_path)
            
        # train_ds = GLMTrainDataset(self.args, cb_train_sam, self.nid2index,  self.nindex2vec)
        # train_dl = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        # for cnt, batch_sample in enumerate(train_dl):
        #     news_index, label = batch_sample
        #     self.update(topics = None, items=news_index, rewards=label, mode = 'item-linear')
    
    def getInv(self, old_Minv, nfv):
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
        # tmp_a = old_Minv.dot(nfv).dot(nfv.T).dot(old_Minv)
        tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
        new_Minv=old_Minv-tmp_a/tmp_b
        return new_Minv

    @torch.no_grad()
    def _get_topic_embs(self):
        self.topic_model.eval() # disable dropout
        self.topic_embs = self.topic_model.get_topic_embeddings_byindex(self.topic_order) #.cpu().numpy()
        
    def update_data_buffer(self, pos, neg, uid, t): 
        self.data_buffer.append([pos, neg, self.clicked_history[uid], uid, t])
        self.data_buffer_lr.append([pos, neg, self.clicked_history[uid], uid, t])

    def construct_trainable_samples_lr(self):
        """construct trainable samples which will be used in NRMS model training
        from self.h_contexts, self.h_actions, self.h_rewards
        """
        tr_samples = []
        tr_rewards = []
        tr_users = []
        # print('Debug self.data_buffer: ', self.data_buffer)

        for i, l in enumerate(self.data_buffer_lr):
            poss, negs, his, uid, tsp = l
            # tr_users.append(uid)
            if len(poss) > 0:
                tr_samples.extend(poss)
                tr_rewards.extend([1] * len(poss))
                tr_neg_len = int(min(1.5 * len(poss), len(negs)))
                tr_samples.extend(np.random.choice(negs, size = tr_neg_len, replace=False))
                tr_rewards.extend([0] * tr_neg_len)
                tr_users.extend([uid]*(len(poss) + tr_neg_len))
        # print('Debug tr_samples: ', tr_samples)
        # print('Debug tr_rewards: ', tr_rewards)
        # print('Debug self.data_buffer: ', self.data_buffer)
        return tr_samples, tr_rewards, tr_users
    
    def train_glm(self, ft_sam, mode='item', save_path=None):
        # TODO: add evaluate
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings[]
            self._get_news_embs(topic=True)
        if len(self.topic_embs) < 1:
            self._get_topic_embs()
        epochs = 2 # self.args.epochs
        
        
        if mode == 'item':
            ft_ds = GLMTrainDataset(samples=ft_sam, nid2index=self.nid2index, uid2index=self.uid2index, nid2topicindex=self.nid2topicindex)
            ft_dl = DataLoader(ft_ds, batch_size=self.args.batch_size, shuffle=True)
            # num_workers=self.args.num_workers
            
            optimizer = optim.Adam(self.lr_model.parameters(), lr=self.args.glm_lr)
            self.lr_model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                ft_loader = tqdm(ft_dl)
                for cnt, batch_sample in enumerate(ft_loader):
                    candidate_news_index, label, uindexs = batch_sample
                    x = self.news_embs[0][candidate_news_index.ravel()] # n_tr, n_dim
                    # uids = [''.join(str(e) for e in uid) for uid in uids]
                    uids = [self.uindex2id[n.item()] for n in uindexs.ravel()]
                    z = np.array([self._get_user_embs(uid, 0) for uid in uids])
                    z = z.reshape(-1, z.shape[-1]) # n_tr, n_dim
                    x = torch.Tensor(x).to(self.device)
                    z = torch.Tensor(z).to(self.device)
                    label = label.ravel().to(self.device)
                    
                    preds = self.lr_model(x, z).ravel()
                    # print('Debug preds: ', preds)
                    loss = self.criterion(preds, label)
                    epoch_loss += loss.detach().cpu().numpy()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if cnt % 10 == 0:
                        ft_loader.set_description(f"[{cnt}]steps item loss: {epoch_loss / (cnt+1):.4f} ")
                        ft_loader.refresh() 
                    
                # self.writer.add_scalars('Training item Loss',
                #     {'Training item': loss}, 
                #     self.round)
                # print('Debug Loss: item - ', loss)
                torch.save(self.lr_model.state_dict(), save_path)
            print('glm item model save to {}'.format(save_path))

        elif mode == 'topic':
            ft_ds = GLMTrainDataset(samples=ft_sam, nid2index=self.nid2index, uid2index=self.uid2index)
            ft_dl = DataLoader(ft_ds, batch_size=self.args.batch_size, shuffle=True)
            # num_workers=self.args.num_workers
            
            optimizer = optim.Adam(self.lr_model_topic.parameters(), lr=self.args.glm_lr)
            
            self.lr_model_topic.train()
            for epoch in range(epochs):
                epoch_loss=0
                ft_loader = tqdm(ft_dl)
                for cnt, batch_sample in enumerate(ft_loader):
                    candidate_news_index, label, uindexs = batch_sample
                    # print('Debug in proposed candidate_news_index: ', candidate_news_index)
                    topic_indexs = [self.nid2topicindex[self.index2nid[n.item()]] for n in candidate_news_index.ravel()]
                    # topic_indexs = []
                    # for n in candidate_news_index.ravel():
                    #     print('Debug n: ', n)
                    #     print('Debug self.index2nid[n.item()]: ', self.index2nid[n.item()])
                    #     index = self.nid2topicindex[self.index2nid[n.item()]] 
                    #     topic_indexs = [index]
                    # print('Debug topic indexs: ', topic_indexs)
                    x = self.topic_embs[topic_indexs].to(self.device) # n_tr, n_dim
                    # uids = [''.join(str(e) for e in uid) for uid in uids]
                    uids = [self.uindex2id[n.item()] for n in uindexs.ravel()]
                    z = np.array([self._get_topic_user_embs(uid, 0).detach().cpu().numpy() for uid in uids])
                    z = torch.Tensor(z.reshape(-1, z.shape[-1])).to(self.device) # n_tr, n_dim
                    label = torch.Tensor(label).ravel().to(self.device)
                
                    preds = self.lr_model_topic(x, z).ravel()
                    # print('Debug labels: ', ft_labels)
                    loss = self.criterion(preds, label)
                    epoch_loss += loss.detach().cpu().numpy()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if cnt % 10 == 0:
                        ft_loader.set_description(f"[{cnt}]steps topic loss: {epoch_loss / (cnt+1):.4f} ")
                        ft_loader.refresh() 
                # debug glm
                # self.writer.add_scalars('Training topic Loss',
                #         {'Training topic': loss}, 
                #         self.round)
                # print('Debug Loss: topic - ', loss)
                torch.save(self.lr_model.state_dict(), save_path)
            print('glm item model save to {}'.format(save_path))
            
    
    def train_lr(self, uid_input=None, mode='item'):
        ft_sam, ft_labels, ft_users = self.construct_trainable_samples_lr()
        ft_labels = torch.Tensor(ft_labels).to(self.device)
        if mode == 'item':
            optimizer = optim.Adam(self.lr_model.parameters(), lr=self.args.glm_lr)
            if len(ft_sam) > 0:
                x = self.news_embs[0][ft_sam] # n_tr, n_dim
                z = np.array([self._get_user_embs(uid, 0) for uid in ft_users])
                z = z.reshape(-1, z.shape[-1]) # n_tr, n_dim
                x = torch.Tensor(x).to(self.device)
                z = torch.Tensor(z).to(self.device)
                self.lr_model.train()
                for epoch in range(self.args.epochs):
                    preds = self.lr_model(x, z).ravel()
                    # print('Debug labels: ', ft_labels)
                    loss = self.criterion(preds, ft_labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                # debug glm
                self.writer.add_scalars('Training item Loss',
                        {'Training item': loss}, 
                        self.round)
                print('Debug Loss: item - ', loss)
            else:
                print('Skip update cb item GLM learner due to lack valid samples!')
        elif mode == 'topic':
            optimizer = optim.Adam(self.lr_model_topic.parameters(), lr=self.args.glm_lr)
            if len(ft_sam) > 0:
                topic_indexs = [self.nid2topicindex[self.index2nid[n]] for n in ft_sam]
                x = self.topic_embs[topic_indexs].to(self.device) # n_tr, n_dim
                z = np.array([self._get_topic_user_embs(uid, 0).detach().cpu().numpy() for uid in ft_users])
                z = torch.Tensor(z.reshape(-1, z.shape[-1])).to(self.device) # n_tr, n_dim
                
                self.lr_model_topic.train()
                for epoch in range(self.args.epochs):
                    preds = self.lr_model_topic(x, z).ravel()
                    # print('Debug labels: ', ft_labels)
                    loss = self.criterion(preds, ft_labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                  # debug glm
                self.writer.add_scalars('Training topic Loss',
                        {'Training topic': loss}, 
                        self.round)
                print('Debug Loss: topic - ', loss)
            else:
                print('Skip update cb topic GLM learner due to lack valid samples!')
        

    def update(self, topics, items, rewards, mode = 'item', uid = None):
        """Updates the posterior using linear bayesian regression formula.
        
        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`/'item-linear'
        """
        print('size of data_buffer: {}; data_buffer_lr: {}'.format(len(self.data_buffer), len(self.data_buffer_lr)))

        if mode == 'item-linear':
            print('Update glmucb topic and item parameters')

            # item
            x = self.news_embs[0][items].T # n_dim, n_hist
            # Update parameters
            self.A+=x.dot(x.T) # n_dim, n_dim
            self.b+=x.dot(rewards) # n_dim, 
            for i in x.T:
                self.Ainv=self.getInv(self.Ainv,i) # n_dim, n_dim
            self.train_lr(mode='item')

            # topic
            x = self.topic_embs[topics].T.cpu().numpy() # n_dim, n_hist
            # Update parameters
            self.A_topic+=x.dot(x.T) # n_dim, n_dim
            self.b_topic+=x.dot(rewards) # n_dim, 
            for i in x.T:
                self.Ainv_topic=self.getInv(self.Ainv_topic,i) # n_dim, n_dim
            self.train_lr(mode='topic')
        elif mode == 'item' or mode == 'topic':
            self.train(mode)
            if self.args.reset_buffer:
                 # REVIEW: only reset buffer ever update_period round
                self.data_buffer_lr = []
        else:
            raise NotImplementedError


    def item_rec(self, uid, cand_news, m = 1): 
        """From neural_linear NeuralGLMAddUCB
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings[]

        z = torch.Tensor(self._get_user_embs(uid, 0)).to(self.device) # (1,d)
        X = torch.Tensor(self.news_embs[0][cand_news]).to(self.device) # (n,d)

        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model.eval()
        mean = self.lr_model.forward(X, z).detach().cpu().numpy().reshape(X.shape[0],)
        X = X.cpu().numpy()
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 
    
    @torch.no_grad()
    def topic_rec(self, uid, m = 1): 
        """
        Args:
            uid: str, a user id 
            m: int, number of items to rec 
        Return: 
            list, containing m element, where each element is a list of cand news index inside a topic (topic can be newly formed if we dynamically form topics)
        """
        # Debug
        self.round +=1
        if len(self.news_embs) < 1:
            self._get_news_embs(topic=True) # init news embeddings[]
        if len(self.topic_embs) < 1:
            self._get_topic_embs()

        z = self._get_topic_user_embs(uid, 0) #.cpu().numpy() # (1,d)
        X = self.topic_embs

        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model_topic.eval()
        mean = self.lr_model_topic.forward(X, z).detach().cpu().numpy().reshape(X.shape[0],)
        X = X.cpu().numpy()
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        sorted_topic_indexs = np.argsort(ucb)[::-1].tolist() # (len(uids),)
        recs = self.topic_cand_news_prep(sorted_topic_indexs,m)
        return recs

    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Reset the CB learner to its initial state (do this for each experiment). """
        # TODO: save and reload
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        self.data_buffer_lr = [] # for logistic regression
        
        self.A =  np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))
        self.lr_model = LogisticRegressionAddtive(self.dim, 1).to(self.device)
        
        # topic lr
        self.A_topic =  np.identity(n=self.dim)
        self.Ainv_topic = np.linalg.inv(self.A)
        self.b_topic = np.zeros((self.dim))
        self.lr_model_topic = LogisticRegressionAddtive(self.dim, 1).to(self.device)
        
        if self.args.pretrain_glm:
            self.pretrain_glm_learner(trial=e,mode='topic')
            self.pretrain_glm_learner(trial=e,mode='item')
        
        
        

class Two_NeuralGBiLinUCB(Two_NeuralGLMAddUCB):
    def __init__(self, args, device, name='2_NeuralGBiLinUCB'):
        """Use Two stage NRMS model with logistic regression (disjoint model for each user)
        """      
        super(Two_NeuralGBiLinUCB, self).__init__(args, device, name)
        
        self.A =  np.identity(n=self.dim**2)
        self.Ainv = np.linalg.inv(self.A)
        self.lr_model = LogisticRegressionBilinear(self.dim, self.dim, 1).to(self.device)
        
        # topic 
        self.A_topic =  np.identity(n=self.dim**2)
        self.Ainv_topic = np.linalg.inv(self.A)
        self.lr_model_topic = LogisticRegressionBilinear(self.dim, self.dim, 1).to(self.device)
        
        self.A = torch.Tensor(self.A).to(self.device)
        self.Ainv = torch.Tensor(self.Ainv).to(self.device)
        self.A_topic = torch.Tensor(self.A_topic).to(self.device)
        self.Ainv_topic = torch.Tensor(self.Ainv_topic).to(self.device)
        
    
    def getInv(self, old_Minv, nfv):
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=torch.outer((old_Minv @ nfv).ravel(),nfv) @ old_Minv
        # tmp_a = old_Minv.dot(nfv).dot(nfv.T).dot(old_Minv)
        tmp_b=1+ nfv.T @ old_Minv @ nfv
        new_Minv=old_Minv-tmp_a/tmp_b
        return new_Minv
    
    def update(self, topics, items, rewards, mode = 'item', uid = None):
        """Updates the posterior using linear bayesian regression formula.
        
        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`/'item-linear'
        """
        print('size of data_buffer: {}; data_buffer_lr: {}'.format(len(self.data_buffer), len(self.data_buffer_lr)))

        if mode == 'item-linear': 
            # item
            print('Update glmucb item parameters')
            t1 = datetime.datetime.now()
            
            # X = self.news_embs[0][items] # n1, n_hist
            # z = self._get_user_embs(uid, 0)[0] # 1,d2
            # for x in X:
            #     vec = np.outer(x.T,z).reshape(-1,) # d1d2,

            #     # Update parameters
            #     self.A+=np.outer(vec, vec.T) # d1d2, d1d2                
            #     self.Ainv=self.getInv(self.Ainv,vec) # n_dim, n_dim
            
            X = torch.Tensor(self.news_embs[0][items]).to(self.device) # n1, n_hist
            z = torch.Tensor(self._get_user_embs(uid, 0)[0]).to(self.device) # 1,d2
            for x in X:
                vec = torch.outer(x.T,z.ravel()).ravel() # d1d2,

                # Update parameters
                self.A+=torch.outer(vec, vec) # d1d2, d1d2                
                self.Ainv=self.getInv(self.Ainv,vec) # n_dim, n_dim
            
            t2 = datetime.datetime.now()
            # print('Debug update item A, Ainv:, ', t2-t1)
            self.train_lr(mode='item')
            t3 = datetime.datetime.now()
            # print('Debug train item lr:, ', t3-t2)

            # topic
            print('Update glmucb topic parameters')

            # X_topic = self.topic_embs[topics].cpu().numpy() # n_dim, n_hist
            # # Update parameters
            # z_topic = self._get_topic_user_embs(uid, 0).detach().cpu().numpy().reshape(1,-1) # 1,d2
            # for x in X_topic:
            #     vec = np.outer(x.T,z_topic).reshape(-1,) # d1d2,

            #     # Update parameters
            #     self.A_topic+=np.outer(vec, vec.T) # d1d2, d1d2                
            #     self.Ainv_topic=self.getInv(self.Ainv_topic,vec) # n_dim, n_dim

            X_topic = self.topic_embs[topics] # n_dim, n_hist
            # Update parameters
            z_topic = self._get_topic_user_embs(uid, 0).reshape(1,-1) # 1,d2
            for x in X_topic:
                vec = torch.outer(x.T,z_topic.ravel()).ravel() # d1d2,

                # Update parameters
                self.A_topic+=torch.outer(vec, vec) # d1d2, d1d2                
                self.Ainv_topic=self.getInv(self.Ainv_topic,vec) # n_dim, n_dim

            t4 = datetime.datetime.now()
            # print('Debug update topic A, Ainv:, ', t4-t3)
            self.train_lr(mode='topic')
            t5 = datetime.datetime.now()
            # print('Debug train item lr:, ', t5-t4)
        elif mode == 'item' or mode == 'topic':
            self.train(mode)
            if self.args.reset_buffer:
                 # REVIEW: only reset buffer ever update_period round
                self.data_buffer_lr = []
        else:
            raise NotImplementedError
        
    def item_rec(self, uid, cand_news, m = 1): 
        """From neural_bilinear NeuralGBiLinUCB
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings[]
        t1 = datetime.datetime.now()
        z = self._get_user_embs(uid, 0).reshape(1,-1) # (1,d)
        X = self.news_embs[0][cand_news] # (n,d)
        cands = np.array([np.outer(x,z) for x in X]).reshape(len(cand_news),-1)
        t2 = datetime.datetime.now()
        # print('Debug news,user inference and get cands:, ', t2-t1)
        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model.eval()
        mean = self.lr_model.forward(
            torch.Tensor(X).to(self.device), 
            torch.Tensor(np.repeat(z, len(cands), axis = 0)).to(self.device)
            ).detach().cpu().numpy().reshape(len(cands),)
        t3 = datetime.datetime.now()
        # print('Debug get ucb mean:, ', t3-t2)

        CI = []
        # Ainv = torch.unsqueeze(torch.Tensor(self.Ainv).to(self.device),dim=0) # 1,4096,4096
        Ainv = torch.unsqueeze(self.Ainv,dim=0) # 1,4096,4096
        cands = torch.unsqueeze(torch.Tensor(np.array(cands)).to(self.device),dim=1) # 5000,1,4096
        CI = torch.bmm(torch.bmm(cands, Ainv.expand(cands.shape[0],-1,-1)),torch.transpose(cands, 1,2)).ravel().cpu().numpy()

        t4 = datetime.datetime.now()
        # print('Debug get ucb CI (on GPU):, ', t4-t3)
        ucb = mean + self.gamma * np.sqrt(CI) # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 
    
    @torch.no_grad()
    def topic_rec(self, uid, m = 1): 
        """
        Args:
            uid: str, a user id 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs(topic=True) # init news embeddings[]
        if len(self.topic_embs) < 1:
            self._get_topic_embs()
            
        t1 = datetime.datetime.now()
        z = self._get_topic_user_embs(uid, 0).reshape(1,-1) #.cpu().numpy() # (1,d) # 
        X = self.topic_embs

        cands = torch.cat([torch.outer(x,z.ravel()) for x in X]).reshape(X.shape[0],-1)
        t2 = datetime.datetime.now()
        # print('Debug news,user inference and get cands:, ', t2-t1)
        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model_topic.eval()
        mean = self.lr_model_topic.forward(
            X, 
            z.repeat(len(cands),1)
            ).detach().cpu().numpy().reshape(len(cands),)
        t3 = datetime.datetime.now()
        # print('Debug get ucb mean:, ', t3-t2)

        CI = []
        # Ainv_topic = torch.unsqueeze(torch.Tensor(self.Ainv_topic).to(self.device),dim=0) # 1,4096,4096
        Ainv_topic = torch.unsqueeze(self.Ainv_topic,dim=0) # 1,4096,4096
        cands = torch.unsqueeze(cands,dim=1) # 5000,1,4096
        CI = torch.bmm(torch.bmm(cands, Ainv_topic.expand(cands.shape[0],-1,-1)),torch.transpose(cands, 1,2)).ravel().cpu().numpy()

        t4 = datetime.datetime.now()
        # print('Debug get ucb CI (on GPU):, ', t4-t3)
        ucb = mean + self.gamma * np.sqrt(CI) # n_cand, 

        sorted_topic_indexs = np.argsort(ucb)[::-1].tolist() # (len(uids),)
        recs = self.topic_cand_news_prep(sorted_topic_indexs,m)
        return recs
    
    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        self.data_buffer_lr = [] # for logistic regression
        
        self.A =  np.identity(n=self.dim**2)
        self.Ainv = np.linalg.inv(self.A)
        self.lr_model = LogisticRegressionBilinear(self.dim, self.dim, 1).to(self.device)
        
        
        self.A_topic =  np.identity(n=self.dim**2)
        self.Ainv_topic = np.linalg.inv(self.A)
        self.lr_model_topic = LogisticRegressionBilinear(self.dim, self.dim, 1).to(self.device)

        self.A = torch.Tensor(self.A).to(self.device)
        self.Ainv = torch.Tensor(self.Ainv).to(self.device)
        self.A_topic = torch.Tensor(self.A_topic).to(self.device)
        self.Ainv_topic = torch.Tensor(self.Ainv_topic).to(self.device)

        if self.args.pretrain_glm:
            self.pretrain_glm_learner(trial=e,mode='topic',model_name = 'bilinear')
            self.pretrain_glm_learner(trial=e,mode='item',model_name = 'bilinear')