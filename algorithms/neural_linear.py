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

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.neural_greedy import NeuralGreedy
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from algorithms.lr_model import LogisticRegression, LogisticRegressionAddtive
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, GLMTrainDataset, load_word2vec, load_cb_topic_news,load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset
from torch.utils.tensorboard import SummaryWriter
import os
from scipy.optimize import newton, root 


class NeuralLinUCB(NeuralGreedy):
    def __init__(self, args, device, name='NeuralLinUCB'):
        """Use NRMS model (disjoint model for each user)
        """      
        super(NeuralLinUCB, self).__init__(args, device, name)
        self.gamma = self.args.gamma
        self.dim = self.args.news_dim

        self.theta = {} # key: uid, value: theta_u
        self.A = {}
        self.Ainv = {}
        self.b = {}

        # debug glm
        out_folder = os.path.join(args.result_path, 'runs') # store final results
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # out_path = os.path.join(out_folder, args.algo_prefix)
        # self.writer = SummaryWriter(out_path) # https://pytorch.org/docs/stable/tensorboard.html

    def pretrain_glm_learner(self, uid=None):
        # NOTE: This intened to pre-train linear models with cb train data. However, we cannot do that since we remove the cb simulated users from the cb train data, and the linear models are disjoint, i.e. theta_u for per user. We can do this for our proposed shared model though.
        """pretrain the generalised linear models (if any) with the pretrained data, to initialise glm parameters.
        """
        print('pretrain glm learner using {} for user {}'.format(self.pretrain_path, uid))
        with open(self.pretrain_path, "rb") as f:
            cb_train_sam = pickle.load(f)

        train_ds = GLMTrainDataset(args, cb_train_sam, self.nid2index,  self.nindex2vec)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        for cnt, batch_sample in enumerate(train_loader):
            news_index, label = batch_sample
            self.update(topics = None, items=news_index, rewards=label, mode = 'item-linear', uid = uid)
        
    def getInv(self, old_Minv, nfv):
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
        # tmp_a = old_Minv.dot(nfv).dot(nfv.T).dot(old_Minv)
        tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
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
        print('size(data_buffer): {}'.format(len(self.data_buffer)))
        if mode == 'item':
            self.train() 

        if mode == 'item-linear':
            print('Update linucb parameters for user {}!'.format(uid))
            x = self.news_embs[0][items].T # n_dim, n_hist
            # Update parameters
            print('Debug x shape: ', x.shape)
            print('Debug A shape: ', self.A[uid].shape)
            self.A[uid]+=x.dot(x.T) # n_dim, n_dim
            self.b[uid]+=x.dot(rewards) # n_dim, 
            for i in x.T:
                self.Ainv[uid]=self.getInv(self.Ainv[uid],i) # n_dim, n_dim
            self.theta[uid]=np.dot(self.Ainv[uid], self.b[uid]) # n_dim,


    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        # ser_vec = self._get_user_embs(uid, 0) # (1,d)
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings
        X = self.news_embs[0][cand_news] # (n,d)

        if uid not in self.A:       
            self.A[uid] = np.identity(n=self.dim)
            self.Ainv[uid] = np.linalg.inv(self.A[uid])
            self.b[uid] = np.zeros((self.dim)) 
            if len(self.clicked_history[uid]) > 0 and not self.args.random_init:
                self.update(topics = None, items=self.clicked_history[uid], rewards=np.ones((len(self.clicked_history[uid]),)), mode = 'item-linear', uid = uid)
            # if self.args.pretrained_mode:
            #     self.pretrain_glm_learner(uid)
            else:
                print('Debug init linear parameters theta randomly')
                self.theta[uid] = np.random.rand(self.dim)
                # self.theta[uid] = np.zeros(self.dim)
                # print('No history of user {}, init theta randomly!'.format(uid))
            
        mean = X.dot(self.theta[uid]) # n_cand, 
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv[uid]).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    # debug glm
    def predict(self, uid, cand_news):
        X = self.news_embs[0][cand_news] # (n,d)
        preds = X.dot(self.theta[uid])
        return preds

    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Save and Reset the CB learner to its initial state (do this for each trial/experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        # self.D = defaultdict(list) 
        # self.c = defaultdict(list)
        self.theta = {}
        self.A = {}
        self.Ainv = {}
        self.b = {}

class NeuralGLMUCB(NeuralLinUCB):
    def __init__(self, args, device, name='NeuralGLMUCB'):
        """Use NRMS model (disjoint model for each user)
        """      
        super(NeuralGLMUCB, self).__init__(args, device, name)
        self.name = name 
        self.lr_models = {} # key: user, value: LogisticRegression(self.dim, 1)
        self.criterion = torch.nn.BCELoss()
        self.data_buffer_lr = [] # for logistic regression

    def set_clicked_history(self, init_clicked_history):
        """
        Args:
            init_click_history: list of init clicked history nindexes
        """
        self.clicked_history = init_clicked_history
        for uid, pos in self.clicked_history.items():
            self.data_buffer_lr.append([pos, [], uid])
        self.init_data_buffer_lr_len = len(self.data_buffer_lr)

    def update_data_buffer(self, pos, neg, uid, t): 
        self.data_buffer.append([pos, neg, self.clicked_history[uid], uid, t])
        self.data_buffer_lr.append([pos, neg, uid])

    def update(self, topics, items, rewards, mode = 'item', uid = None):
        """Updates the posterior using linear bayesian regression formula.
        
        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`/'item-linear'
        """
        print('size(data_buffer): {}'.format(len(self.data_buffer)))
        if mode == 'item':
            self.train() 

        if mode == 'item-linear':
            print('Update glmucb parameters for user {}!'.format(uid))
            x = self.news_embs[0][items].T # n_dim, n_hist
            # Update parameters
            self.A[uid]+=x.dot(x.T) # n_dim, n_dim
            for i in x.T:
                self.Ainv[uid]=self.getInv(self.Ainv[uid],i) # n_dim, n_dim
            self.train_lr(uid)
    
    def construct_trainable_samples_lr(self, tr_uid):
        """construct trainable samples which will be used in NRMS model training
        from self.h_contexts, self.h_actions, self.h_rewards
        """
        tr_samples = []
        tr_rewards = []
        # print('Debug self.data_buffer_lr: ', self.data_buffer_lr)

        for i, l in enumerate(self.data_buffer_lr):
            poss, negs, uid = l
            if uid == tr_uid and len(poss) > 0:
                tr_samples.extend(poss)
                tr_rewards.extend([1] * len(poss))
                tr_neg_len = int(min(1.5 * len(poss), len(negs)))
                tr_samples.extend(np.random.choice(negs, size = tr_neg_len, replace=False))
                tr_rewards.extend([0] * tr_neg_len)
            # tr_samples.extend(poss)
            # tr_rewards.extend([1]*len(poss))
            # tr_samples.extend(negs)
            # tr_rewards.extend([0]*len(negs))

            # self.data_buffer_lr.remove(l)
        # print('Debug tr_samples: ', tr_samples)
        # print('Debug tr_rewards: ', tr_rewards)
        # print('Debug self.data_buffer: ', self.data_buffer)
        return np.array(tr_samples), np.array(tr_rewards)
        
    def train_lr(self, uid):
        # TODO: use GLMTrainDataset
        optimizer = optim.Adam(self.lr_models[uid].parameters(), lr=self.args.glm_lr)
        ft_sam, ft_labels = self.construct_trainable_samples_lr(uid)
        if len(ft_sam) > 0:
            
            x = self.news_embs[0][ft_sam] # n_tr, n_dim
            self.lr_models[uid].train()
            # REVIEW: for epoch in range(self.args.epochs):
            preds = self.lr_models[uid](torch.Tensor(x)).ravel()
            # print('Debug preds: ', preds)
            # print('Debug labels: ', rewards)
            loss = self.criterion(preds, torch.Tensor(ft_labels))
            print('Debug for uid {} loss {} '.format(uid, loss))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # debug glm
            # self.writer.add_scalars('{} Training Loss'.format(uid),
            #         {'Training': loss}, 
            #         len(self.data_buffer_lr) - self.init_data_buffer_lr_len)

    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings
        X = self.news_embs[0][cand_news] # (n,d)

        if uid not in self.A:       
            self.A[uid] = np.identity(n=self.dim)
            self.Ainv[uid] = np.linalg.inv(self.A[uid])
            self.lr_models[uid] = LogisticRegression(self.dim, 1)
            if len(self.clicked_history[uid]) > 0 and not self.args.random_init:
                print('Debug init lr parameters by clicked history ({})'.format(self.init_data_buffer_lr_len))
                self.train_lr(uid) # update by data_buffer_lr
            
        self.lr_models[uid].eval()
        mean = self.lr_models[uid].forward(torch.Tensor(X)).detach().numpy().reshape(X.shape[0],) # n_cand, 
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv[uid]).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    # debug glm
    def predict(self, uid, cand_news):
        X = self.news_embs[0][cand_news] # (n,d)
        preds = self.lr_models[uid].forward(torch.Tensor(X)).detach().numpy().reshape(X.shape[0],) # n_cand, 
        return preds


    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] 
        self.data_buffer_lr = [] 
        # self.D = defaultdict(list) 
        # self.c = defaultdict(list)
        self.lr_models = {}
        self.A = {}
        self.Ainv = {}

class NeuralGLMUCB_LBFGS(NeuralGLMUCB):
    def __init__(self, args, device, name='NeuralGLMUCB_LBFGS'):
        """Use NRMS model (disjoint model for each user)
        """      
        super(NeuralGLMUCB_LBFGS, self).__init__(args, device, name)

    def train_lr(self, uid):
        optimizer = optim.LBFGS(self.lr_models[uid].parameters(), lr=self.args.glm_lr)
        ft_sam, ft_labels = self.construct_trainable_samples_lr(uid)
        if len(ft_sam) > 0:
            x = self.news_embs[0][ft_sam] # n_tr, n_dim         
            def closure():   
                self.lr_models[uid].train()
                # REVIEW: for epoch in range(self.args.epochs):
                preds = self.lr_models[uid](torch.Tensor(x)).ravel()
                # print('Debug preds: ', preds)
                # print('Debug labels: ', rewards)
                loss = self.criterion(preds, torch.Tensor(ft_labels))
                print('Debug for uid {} loss {} '.format(uid, loss))
                optimizer.zero_grad()
                loss.backward()

                # debug glm
                # self.writer.add_scalars('{} Training Loss'.format(uid),
                #         {'Training': loss}, 
                #         len(self.data_buffer_lr) - self.init_data_buffer_lr_len)
                return loss
            optimizer.step(closure)

            

class NeuralGLMUCB_Newton(NeuralGLMUCB):
    def __init__(self, args, device, name='NeuralGLMUCB_Newton'):
        """Use NRMS model (disjoint model for each user)
        """      
        super(NeuralGLMUCB_Newton, self).__init__(args, device, name)
        self.name = name 
        self.theta = {} # key: user, value: theta
        self.func = {} # key: user, value: func
        self.jaco = {} # key: user, value: jocobian 

    def update_data_buffer(self, pos, neg, uid, t): 
        self.data_buffer.append([pos, neg, self.clicked_history[uid], uid, t])
        self.data_buffer_lr.append([pos, neg, self.clicked_history[uid], uid, t])

    def update(self, topics, items, rewards, mode = 'item', uid = None):
        """Updates the posterior using linear bayesian regression formula.
        
        Args:
            topics: list of `rsec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`/'item-linear'
        """
        print('size(data_buffer): {}'.format(len(self.data_buffer)))
        if mode == 'item':
            self.train() 

        if mode == 'item-linear':
            print('Update glmucb parameters for user {}!'.format(uid))
            x = self.news_embs[0][items] # n_hist, n_dim
            # def sigmoid(y):
            #     if len(y.shape) > 1:
            #         assert y.shape[0] == 1
            #         y = y[0]
            #     return np.array([1/(1 + math.exp(-i)) for i in y])
            # # Update parameters
            self.A[uid]+=x.T.dot(x) # n_dim, n_dim
            for i in x:
                self.Ainv[uid]=self.getInv(self.Ainv[uid],i) # n_dim, n_dim

            def fun(theta,x,rewards):
                # print('Debug self.func[uid] shape, ', self.func[uid].shape)
                # print('Debug theta shape: ', theta.shape)
                # print('Debug x shape: ', x.shape)
                # print('Debug (rewards - sigmoid(theta @ x.T)).reshape(1,-1)', (rewards - sigmoid(theta @ x.T)).reshape(1,-1).shape)
                return self.func[uid] + (rewards - torch.sigmoid(theta @ x.T)).reshape(1,-1) @ x # 1, n_dim
            # def jac(theta):
            #     output = self.jaco[uid]
            #     for i in x:
            #         i = i.reshape(1,-1)
            #         output+= (sigmoid(theta @ i.T) - 1) * i.T@i
            #     print('Debug jac func output shape: ', output.shape)
            #     return output

            self.theta[uid].requires_grad = True
            optimizer = optim.LBFGS([self.theta[uid]], lr=self.args.glm_lr)
            optimizer.zero_grad()
            loss = fun(self.theta[uid], torch.Tensor(x), torch.Tensor(rewards))
            loss.mean().backward()
            optimizer.step(lambda: fun(self.theta[uid], torch.Tensor(x), torch.Tensor(rewards)))

            # self.theta[uid] = root(fun, self.theta[uid], method = 'krylov').x
            self.func[uid] = fun(self.theta[uid])
            # self.jaco[uid] = jac(self.theta[uid])
            # print('Debug theta uid shape: ', self.theta[uid].shape)
            # print('Debug self.func[uid] shape: ', self.func[uid].shape)
            # print('Debug self.jaco[uid] shape: ', self.jaco[uid].shape)

            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html#scipy.optimize.root
            
            # TODO: in glm add pre-trained using clicked history, as did in linucb

    def train_lr(self, uid):
        ft_sam, ft_labels = self.construct_trainable_samples_lr(uid)
        if len(ft_sam) > 0:
            x = self.news_embs[0][ft_sam] # n_tr, n_dim
        func = (ft_labels - self.sigmoid(x.T@ self.theta[uid]))@ x
        newton(func, 0, )


    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings
        X = self.news_embs[0][cand_news] # (n,d)

        if uid not in self.A:       
            self.A[uid] = np.identity(n=self.dim)
            self.Ainv[uid] = np.linalg.inv(self.A[uid])
            self.func[uid] = torch.Tensor(np.zeros((1,self.dim)))
            # self.jaco[uid] = np.zeros((self.dim,self.dim))
            self.theta[uid] = torch.Tensor(np.zeros((1,self.dim)))
            if len(self.clicked_history[uid]) > 0:
                self.update(topics = None, items=self.clicked_history[uid], rewards=np.ones((len(self.clicked_history[uid]),)), mode = 'item-linear', uid = uid)
    
        theta = self.theta[uid].reshape(-1,)
        mean = X.dot(theta) # n_cand, 
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv[uid]).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    # debug glm
    def predict(self, uid, cand_news):
        X = self.news_embs[0][cand_news] # (n,d)
        theta = self.theta[uid].reshape(-1,)
        preds = X.dot(theta)
        return preds


    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] 
        self.data_buffer_lr = [] 
        # self.D = defaultdict(list) 
        # self.c = defaultdict(list)
        self.lr_models = {}
        self.A = {}
        self.Ainv = {}


class NeuralGLMAddUCB(NeuralGreedy):
    def __init__(self, args, device, name='neural_glmadducb'):
        """Use NRMS model with logistic regression (user item hybrid)
        \bm{x}_{i}^T \hat{\bm{\theta}}_x +  {\hat{\bm{\theta}}_z}^T \bm{z}_{u}+\alpha  \sqrt{\bm{x}_i^T A_t^{-1} \bm{x}_i},
        """      
        super(NeuralGLMAddUCB, self).__init__(args, device, name)
        self.n_inference = 1
        self.gamma = self.args.gamma
        self.dim = self.args.news_dim
        # self.dim = 400 # self.args.latent_dim

        self.A =  np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))

        self.lr_model = LogisticRegressionAddtive(self.dim, 1)
        self.criterion = torch.nn.BCELoss()
        self.data_buffer_lr = [] # for logistic regression

    def getInv(self, old_Minv, nfv):
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
        # tmp_a = old_Minv.dot(nfv).dot(nfv.T).dot(old_Minv)
        tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
        new_Minv=old_Minv-tmp_a/tmp_b
        return new_Minv

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

            self.data_buffer_lr.remove(l)
        # print('Debug tr_samples: ', tr_samples)
        # print('Debug tr_rewards: ', tr_rewards)
        # print('Debug self.data_buffer: ', self.data_buffer)
        return tr_samples, tr_rewards, tr_users

    def train_lr(self, uid):
        optimizer = optim.Adam(self.lr_model.parameters(), lr=self.args.glm_lr)
        ft_sam, ft_labels, ft_users = self.construct_trainable_samples_lr()
        
        if len(ft_sam) > 0:
            x = self.news_embs[0][ft_sam] # n_tr, n_dim
            z = np.array([self._get_user_embs(uid, 0) for uid in ft_users])
            z = z.reshape(-1, z.shape[-1]) # n_tr, n_dim
            self.lr_model.train()
            for epoch in range(self.args.epochs):
                preds = self.lr_model(torch.Tensor(x), torch.Tensor(z)).ravel()
                # print('Debug labels: ', ft_labels)
                loss = self.criterion(preds, torch.Tensor(ft_labels))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
    def update(self, topics, items, rewards, mode = 'item', uid = None):
        """Updates the posterior using linear bayesian regression formula.
        
        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`/'item-linear'
        """
        print('size(data_buffer): {}'.format(len(self.data_buffer)))
        if mode == 'item':
            self.train() 

        if mode == 'item-linear':
            print('Update linucb parameters for user {}!'.format(uid))
            x = self.news_embs[0][items].T # n_dim, n_hist
            # Update parameters
            self.A+=x.dot(x.T) # n_dim, n_dim
            self.b+=x.dot(rewards) # n_dim, 
            for i in x.T:
                self.Ainv=self.getInv(self.Ainv,i) # n_dim, n_dim

            self.train_lr(uid)


    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings[]

        z = self._get_user_embs(uid, 0) # (1,d)
        X = self.news_embs[0][cand_news] # (n,d)

        # x_mean = X.dot(self.theta_x) + z.dot(self.theta_z)# n_cand, 
        self.lr_model.eval()
        mean = self.lr_model.forward(torch.Tensor(X), torch.Tensor(z)).detach().numpy().reshape(X.shape[0],)
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        
        self.A =  np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))

        self.lr_model = LogisticRegressionAddtive(self.dim, 1)
        self.data_buffer_lr = [] # for logistic regression