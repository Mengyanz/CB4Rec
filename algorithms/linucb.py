"""Define a linear ucb recommendation policy. """

import math 
import numpy as np 
from collections import defaultdict
import torch 
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.neural_greedy import SingleStageNeuralGreedy
from algorithms.lr_model import LogisticRegression
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class SingleStageLinUCB(ContextualBanditLearner):
    def __init__(self, args, device, name='SingleStageLinUCB'):
        """LinUCB.
        """
        super(SingleStageLinUCB, self).__init__(args, device, name)
        
        word2vec = torch.from_numpy(self.word2vec).float()
        self.word_embedding = nn.Embedding.from_pretrained(word2vec, freeze=True).to(self.device)

        self.gamma = self.args.gamma
        self.dim = self.args.word_embedding_dim
        self.theta = {} # key: uid, value: theta_u

        self.A = {}
        self.Ainv = {}
        self.b = {}

    def update_clicked_history(self, pos, uid):
        """
        Args:
            pos: a list of str nIDs, positive news of uid 
            uid: str, user id 
        """
        # DO NOT UPDATE CLICKED HISTORY
        pass 

    def getInv(self, old_Minv, nfv):
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
        # tmp_a = old_Minv.dot(nfv).dot(nfv.T).dot(old_Minv)
        tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
        new_Minv=old_Minv-tmp_a/tmp_b
        return new_Minv

    def update(self, topics, items, rewards, mode = 'topic', uid = None):
        """Update its internal model. 

        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its integer index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`
            uid: user id.
        """
        if mode == 'item':
            print('Update linucb parameters for user {}!'.format(uid))
            x = self._get_news_embedding(items).T # n_dim, n_hist
            # Update parameters
            self.A[uid]+=x.dot(x.T) # n_dim, n_dim
            self.b[uid]+=x.dot(rewards) # n_dim, 
            for i in x.T:
                self.Ainv[uid]=self.getInv(self.Ainv[uid],i) # n_dim, n_dim
            self.theta[uid]=np.dot(self.Ainv[uid], self.b[uid]) # n_dim,
        
    def _get_news_embedding(self, nindexes):
        """word2vec embedding.
        Args
            nindexes: list of nindexes
        Return
            embedding: array with n_items, n_embedding_dim
        """
        vecs = torch.Tensor([self.nindex2vec[nindex] for nindex in nindexes]).to(self.device) # n_item, n_word
        # embedding = self.word_embedding(vecs.long()).view(vecs.shape[0], -1).detach().cpu().numpy() # n_item, n_word * n_word_embedding
        embedding = self.word_embedding(vecs.long()).sum(axis=1).detach().cpu().numpy() # n_item, n_word_embedding
        return embedding

    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        X = self._get_news_embedding(cand_news)
        # print('Debug X shape: ', X.shape)

        if uid not in self.A:
            self.A[uid] = np.identity(n=self.dim)
            self.Ainv[uid] = np.linalg.inv(self.A[uid])
            self.b[uid] = np.zeros((self.dim)) 
            if self.pretrained_mode and len(self.clicked_history[uid]) > 0:
                self.update(topics = None, items=self.clicked_history[uid], rewards=np.ones((len(self.clicked_history[uid]),)), mode = 'item', uid = uid)
            else:
                # REVIEW: alternatively, we can init theta to zeros
                self.theta[uid] = np.random.rand(self.dim)
                # self.theta[uid] = np.zeros(self.dim)
                # print('No history of user {}, init theta randomly!'.format(uid))
        
        mean = X.dot(self.theta[uid]) # n_cand, 
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv[uid]).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 
        

    def reset(self):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] 
        # self.D = defaultdict(list) 
        # self.c = defaultdict(list)
        self.theta = {}
        self.A = {}
        self.Ainv = {}
        self.b = {}

class GLMUCB(SingleStageLinUCB):
    """Single stage Generalised linear model UCB
    We only consider Logistic Regression here.
    """

    def __init__(self, args, device, name='GLMUCB'):
        """GLMUCB.
        """
        super(GLMUCB, self).__init__(args, device, name)
        self.lr_models = {} # key: user, value: LogisticRegression(self.dim, 1)
        self.criterion = torch.nn.BCELoss()

    def construct_trainable_samples(self, tr_uid):
        """construct trainable samples which will be used in NRMS model training
        from self.h_contexts, self.h_actions, self.h_rewards
        """
        tr_samples = []
        tr_rewards = []
        # print('Debug self.data_buffer: ', self.data_buffer)

        for i, l in enumerate(self.data_buffer):
            poss, negs, his, uid, tsp = l
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

            self.data_buffer.remove(l)
        # print('Debug tr_samples: ', tr_samples)
        # print('Debug tr_rewards: ', tr_rewards)
        # print('Debug self.data_buffer: ', self.data_buffer)
        return np.array(tr_samples), np.array(tr_rewards)

    def train_lr(self, uid):
        optimizer = optim.Adam(self.lr_models[uid].parameters(), lr=self.args.lr)
        ft_sam, ft_labels = self.construct_trainable_samples(uid)
        x = self._get_news_embedding(ft_sam) # n_tr, n_dim
        if len(ft_sam) > 0:
            self.lr_models[uid].train()
            for epoch in range(self.args.epochs):
                preds = self.lr_models[uid](torch.Tensor(x))
                # print('Debug preds: ', preds)
                # print('Debug labels: ', rewards)
                loss = self.criterion(preds, torch.Tensor(ft_labels))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def update(self, topics, items, rewards, mode = 'topic', uid = None):
        """Update its internal model. 

        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its integer index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`
            uid: user id.
        """
        if mode == 'item':
            print('Update glmucb parameters for user {}!'.format(uid))
            x = self._get_news_embedding(items).T # n_dim, n_hist
            # Update parameters
            self.A[uid]+=x.dot(x.T) # n_dim, n_dim
            for i in x.T:
                self.Ainv[uid]=self.getInv(self.Ainv[uid],i) # n_dim, n_dim
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
        X = self._get_news_embedding(cand_news)

        if uid not in self.lr_models:
            self.A[uid] = np.identity(n=self.dim)
            self.Ainv[uid] = np.linalg.inv(self.A[uid])
            self.lr_models[uid] = LogisticRegression(self.dim, 1)
            if len(self.clicked_history[uid]) > 0:
                self.update(topics = None, items=self.clicked_history[uid], rewards=np.ones((len(self.clicked_history[uid]),)), mode = 'item', uid = uid)
        
        self.lr_models[uid].eval()
        mean = self.lr_models[uid].forward(torch.Tensor(X)).detach().numpy().reshape(X.shape[0],) # n_cand, 
        CI = np.array([self.gamma * np.sqrt(x.dot(self.Ainv[uid]).dot(x.T)) for x in X])
        ucb = mean + CI # n_cand, 

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        # print('Debug nid_argmax: ', nid_argmax)
        # print('Debug mean: ', mean[nid_argmax])
        # print('Debug CI: ', CI[nid_argmax])
        # print('Debug ucb: ', ucb[nid_argmax])

        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    def reset(self):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] 
        # self.D = defaultdict(list) 
        # self.c = defaultdict(list)
        self.lr_models = {}
        self.A = {}
        self.Ainv = {}