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
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class SingleStageLinUCB(ContextualBanditLearner):
    def __init__(self,device, args, rec_batch_size = 1, gamma = 1, pretrained_mode=True, name='SingleStageLinUCB'):
        """LinUCB.
            Args:
                rec_batch_size: int, recommendation size. 
                gamma: float, parameter that balancing two terms in ucb.
                pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 

        """
        super(SingleStageLinUCB, self).__init__(args, rec_batch_size, pretrained_mode, name)
        self.name = name 
        self.device = device 

        self.nid2index, word2vec, self.nindex2vec = load_word2vec(args)

        word2vec = torch.from_numpy(word2vec).float()
        self.word_embedding = nn.Embedding.from_pretrained(word2vec, freeze=True).to(self.device)


        topic_news = load_cb_topic_news(args) # dict, key: subvert; value: list nIDs 
        cb_news = []
        for k,v in topic_news.items():
            cb_news.append(l.strip('\n').split("\t")[0] for l in v) # get nIDs 
        cb_news = [item for sublist in cb_news for item in sublist]
        # DEBUG:
        print('Warning: for debug, sample 1000 candidates news! Remove this line for full evaluation!')
        self.cb_news = np.random.choice(cb_news, size=1000, replace=False).tolist()

        self.gamma = gamma
        self.dim = 300 # TODO: make it a parameter
        self.theta = {} # key: uid, value: theta_u
        self.D = defaultdict(list) # key: uid, value: list of nindex of uid's interactions
        self.c = defaultdict(list) # key: uid, value: list of labels of uid's interactions

    def update_clicked_history(self, pos, uid):
        """
        Args:
            pos: a list of str nIDs, positive news of uid 
            uid: str, user id 
        """
        # DO NOT UPDATE CLICKED HISTORY
        pass 

    def update_data_buffer(self, pos, neg, uid, t): 
        for nid in pos:
            self.D[uid].append(nid)
            self.c[uid].append(1)
        for nid in neg:
            self.D[uid].append(nid)
            self.c[uid].append(0)
        print('size(data_buffer): {}'.format(len(self.D)))

    def _get_news_embedding(self, nindexes):
        """
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
        X = self._get_news_embedding(cand_news).T
        # print('Debug X shape: ', X.shape)
        
        if uid not in self.theta:
            self.update_data_buffer(self.clicked_history[uid], [], uid, -1)
            self.update(None, None, None)
            

        D = self._get_news_embedding(self.D[uid])
        # print('Debug D shape: ', D.shape)
        inverse_term = np.linalg.inv(D.T.dot(D) + np.identity(D.shape[-1]))
        # print('Debug inverse term shape: ', inverse_term.shape)

        if len(self.clicked_history[uid]) == 0:
            # REVIEW: alternatively, we can init theta to zeros
            self.theta[uid] = np.random.rand(self.dim)
            # self.theta[uid] = np.zeros(self.dim)
            print('No history of user {}, init theta randomly!'.format(uid))
        else:
            self.theta[uid] = inverse_term.dot(D.T).dot(self.c[uid])
            # print('Debug theta shape: ', self.theta[uid].shape)
        
        mean = self.theta[uid].T.dot(X)
        if D.shape[0] == 0:
            ucb = mean
        else:
            CI = np.array([self.gamma * np.sqrt(x.T.dot(inverse_term).dot(x)) for x in X.T])
            ucb = mean + CI

        nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 
        
         # batch_size = min(self.args.max_batch_size, len(cand_news))

        # # get user vect 
     
        # h = self.clicked_history[uid]
        # h = h + [0] * (self.args.max_his_len - len(h))
        # h = self.nindex2vec[h]

        # h = torch.Tensor(h[None,:,:])
        # sed = SimEvalDataset2(self.args, cand_news, self.nindex2vec)
        # rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=4) 

        # cand_vecs = []
        # for cn in rdl:
        #     cand_vec, user_vec = self.model.forward(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, return_embedding = True)
        #     cand_vecs.append(cand_vec.detach().cpu().numpy()) 
        # cand_vecs = np.concatenate(cand_vecs).squeeze(1) # n_cand_news, n_dim (1000, 400)
        # user_vecs = user_vec[0].repeat(cand_vecs.shape[0], 1).detach().cpu().numpy() # n_cand_news, n_dim (1000, 400)
        # context_vecs = np.concatenate([cand_vecs, user_vecs], axis = 1)  # (1000, 800)
        # mean = np.dot(context_vecs, self.theta[uid].T) # (1000,)

    def sample_actions(self, uid): 
        """Choose an action given a context. 
        Args:
            uids: one str uID. 

        Return: 
            topics: (len(uid), `rec_batch_size`)
            items: (len(uid), `rec_batch_size`) 
        """
   
        cand_news = [self.nid2index[n] for n in self.cb_news]
        rec_items = self.item_rec(uid, cand_news, self.rec_batch_size)

        return np.empty(0), rec_items

    def reset(self):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.D = defaultdict(list) 
        self.c = defaultdict(list)
        self.theta = {}