"""Define a simple UCB. """

import math 
import numpy as np 
import pandas as pd
from collections import defaultdict
import torch 
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.neural_greedy import NeuralGreedy, Two_NeuralGreedy
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news,load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class DummyThompsonSampling_NeuralDropoutUCB(ContextualBanditLearner): #@Thanh: for the sake of testing my pipeline only 
    def __init__(self, args, device, name='ThompsonSampling_NeuralDropoutUCB'):
        """Two stage exploration. Use NRMS model. 
        """
        super(DummyThompsonSampling_NeuralDropoutUCB, self).__init__(args, device, name)
        self.n_inference = self.args.n_inference 
        self.pretrained_mode = self.args.pretrained_mode 
        self.name = name 
        self.device = device 

        # preprocessed data 
        self.nid2index, word2vec, self.nindex2vec = load_word2vec(args)
        topic_news = load_cb_topic_news(args) # dict, key: subvert; value: list nIDs 
        cb_news = defaultdict(list)
        for k,v in topic_news.items():
            cb_news[k] = [l.strip('\n').split("\t")[0] for l in v] # get nIDs 
        self.cb_news = cb_news 

        # model 
        self.model = NRMS_Model(word2vec).to(self.device)
 
        self.cb_topics = list(self.cb_news.keys())

        self.alphas = {}
        self.betas = {}

        for topic in self.cb_topics:
            self.alphas[topic] = 1
            self.betas[topic] = 1

    def topic_rec(self):
        """    
        Return
            rec_topic: one recommended topic
        """
        ss =[] 
        for topic in self.active_topics:
            s = np.random.beta(a= self.alphas[topic], b= self.betas[topic])
            ss.append(s)
        rec_topic = self.active_topics[np.argmax(ss)]
        return rec_topic

    def item_rec(self, uid, cand_news): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 

        Return: 
            item: int 
        """
        batch_size = min(self.args.max_batch_size, len(cand_news))

        # get user vect 
     
        h = self.clicked_history[uid]
        h = h + [0] * (self.args.max_his_len - len(h))
        h = self.nindex2vec[h]

        h = torch.Tensor(h[None,:,:])
        sed = SimEvalDataset2(self.args, cand_news, self.nindex2vec)
        #TODO: Use Dataset is clean and good when len(uids) is large. When len(uids) is small, is it faster to not use Dataset?
        rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers) 

        all_scores = []
        for _ in range(self.n_inference):
            scores = []
            for cn in rdl:
                score = self.model.forward(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, compute_loss=False)
                scores.append(score.detach().cpu().numpy()) 
            scores = np.concatenate(scores) 
            all_scores.append(scores) 

        all_scores = np.array(all_scores).squeeze(-1) # (n_inference,len(cand_news))
        mu = np.mean(all_scores, axis=0) 
        std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
        ucb = mu + std # (n,b) 
        nid_argmax = np.argmax(ucb).tolist() 
        return cand_news[nid_argmax] 

    def construct_trainable_samples(self):
        """construct trainable samples which will be used in NRMS model training
        from self.h_contexts, self.h_actions, self.h_rewards
        """
        tr_samples = []
        # for i, l in enumerate(self.h_contexts):
        #     _, _, his, uid, tsp = l
        #     poss = []
        #     negs = []
        #     for j, reward in enumerate(self.h_rewards[i]):
        #         if reward == 1:
        #             poss.append(self.h_actions[i][j])
        #         elif reward == 0:
        #             negs.append(self.h_actions[i][j])
        #         else:
        #             raise Exception("invalid reward")

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
            ft_dl = DataLoader(ft_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            
            # do one epoch only
            loss = 0
            self.model.train()
            # ft_loader = tqdm(ft_dl)
            for cnt, batch_sample in enumerate(ft_dl):
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

    def update(self, topics, items, rewards, mode = 'topic',uid = None):
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
        # Update the topic model 
        if mode == 'topic': 
            for i, topic in enumerate(topics):  # h_actions are topics
                assert rewards[i] in {0,1}
                self.alphas[topic] += rewards[i]
                self.betas[topic] += 1 - rewards[i]

        # Update the user_encoder and news_encoder using `self.clicked_history`
        if mode == 'item': 
            self.train() 

    def sample_actions(self, uid): 
        """Choose an action given a context. 
        Args:
            uid: str, user id

        Return: 
            topics: (`rec_batch_size`)
            items: (`rec_batch_size`) @TODO: what if one topic has less than `rec_batch_size` numbers of items? 
        """
        rec_topics = []
        rec_items = []
        self.active_topics = self.cb_topics.copy()
        while len(rec_items) < self.rec_batch_size:
            rec_topic = self.topic_rec()
            rec_topics.append(rec_topic)
            self.active_topics.remove(rec_topic)

            cand_news = [self.nid2index[n] for n in self.cb_news[rec_topic]]
            # DEBUG
            print('DEBUG:', rec_topic, len(cand_news))
            rec_item = self.item_rec(uid, cand_news)
            rec_items.append(rec_item)
        
        return rec_topics, rec_items
    
class ThompsonSampling_NeuralDropoutUCB(NeuralDropoutUCB):
    def __init__(self, args, device, name='2_ThompsonSampling_NeuralDropoutUCB'):
        """Two stage: Thompson sampling for the topic exploration, neural based dropout ucb for item level exploration.

        Args:
            topic_budget: int
                score budget for topics
            uniform_init: bool
                init TS alpha and beta uniformly
            alphas: dict
                key: topic, value: alpha for topic
            betas: dict
                key: topic, value: beta for topic
        """
        super(ThompsonSampling_NeuralDropoutUCB, self).__init__(args, device, name)
        self.topic_budget = len(self.cb_topics) # the score budget for topic exploration
        self.uniform_init = self.args.uniform_init 
        self.alphas = {}
        self.betas = {}

    def set_clicked_history(self, init_clicked_history):
        """
        Args:
            init_click_history: list of init clicked history nindexs
        """
        self.clicked_history = init_clicked_history
        # if self.uniform_init:
        for topic in self.cb_topics:
            self.alphas[topic] = 1
            self.betas[topic] = 1
        # TODO: set alpha and beta based on clicked history
        # else:
        #     print('Debug non uniform init for ts!')
        #     clicked_nindexs = np.concatenate(list(init_clicked_history.values()))
        #     ave_clicks = len(clicked_nindexs)/len(self.cb_topics)
        #     print('Debug n_clicks all: ', len(clicked_nindexs))
        #     print('Debug ave_clicks over topics: ', ave_clicks)
        #     for topic in self.cb_topics:
        #         topic_nindexs = [self.nid2index[n] for n in self.cb_news[topic]]
        #         self.alphas[topic] = len([nindex for nindex in clicked_nindexs if nindex in set(topic_nindexs)]) # n_clicks of topic in clicked histories
        #         # print('Debug topic {} with alpha init as {}'.format(topic, self.alphas[topic]))
        #         self.betas[topic] = ave_clicks

    def topic_rec(self, m = 1):
        """    
        Return
            rec_topic: one recommended topic
        """

        ss =[] 
        for topic in self.cb_topics:
            s = np.random.beta(a= self.alphas[topic], b= self.betas[topic])
            ss.append(s)
        
        if self.args.dynamic_aggregate_topic:
            sort_topics = np.array(self.cb_topics)[np.array(np.argsort(ss)[::-1])]
            rec_topics = []
            for topic in sort_topics:
                while len(rec_topics) < m:
                    rec_topic = []
                    rec_topic_size = 0
                    if rec_topic_size < self.args.min_item_size:
                        rec_topic_size+= len(self.cb_news[topic])
                        rec_topic.append(topic)
                    else:
                        rec_topics.append(rec_topic)
                        rec_topic = []
                        rec_topic_size = 0
        else:
            rec_topics = np.array(self.cb_topics)[np.array(np.argsort(ss)[::-1][:m])]

        return rec_topics

    def update(self, topics, items, rewards, mode = 'topic', uid = None):
        """Update its internal model. 

        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`
        """
        # Update the topic model 
        if mode == 'topic': 
            for i, topic in enumerate(topics):  # h_actions are topics
                # print("attention::", rewards[i])
                assert rewards[i] in {0,1}
                self.alphas[topic] += rewards[i]
                self.betas[topic] += 1 - rewards[i]

        # Update the user_encoder and news_encoder using `self.clicked_history`
        if mode == 'item': 
            print('size(data_buffer): {}'.format(len(self.data_buffer)))
            self.train() 
            self._get_news_embs()

    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """

        score_budget = self.per_rec_score_budget - int(self.topic_budget/m)
        if len(cand_news)>score_budget:
            print('Randomly sample {} candidates news out of candidate news ({})'.format(score_budget, len(cand_news)))
            cand_news = np.random.choice(cand_news, size=score_budget, replace=False).tolist()

        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings

        all_scores = []           
        for i in range(self.n_inference): 
            user_vecs = self._get_user_embs(uid, i) # (b,d)
            scores = self.news_embs[i][cand_news] @ user_vecs.T # (n,b) 
            all_scores.append(scores) 
        
        all_scores = np.array(all_scores).squeeze(-1)  # (n_inference,n,b)
        mu = np.mean(all_scores, axis=0) 
        std = np.std(all_scores, axis=0) # / math.sqrt(self.n_inference) 
        print('Debug mean: ', mu)
        print('Debug std: ', std)
        ucb = mu + self.gamma * std # (n,) 
        nid_argmax = np.argsort(ucb, axis = 0)[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

    def sample_actions(self, uid): 
        """Choose an action given a context. 
        Args:
            uid: str, user id

        Return: 
            topics: (`rec_batch_size`)
            items: (`rec_batch_size`) @TODO: what if one topic has less than `rec_batch_size` numbers of items? 
        """

        rec_items = []
        rec_topics = self.topic_rec(m = self.rec_batch_size)
        selected_topics = []
        for rec_topic in rec_topics:
            if self.args.dynamic_aggregate_topic:
                cand_news = [self.nid2index[n] for n in self.cb_news[recs] for recs in rec_topic]
            else:
                cand_news = [self.nid2index[n] for n in self.cb_news[rec_topic]]
            print('DEBUG:', rec_topic, len(cand_news))
            rec_item = self.item_rec(uid, cand_news,m=1)[0]
            rec_items.append(rec_item)
            selected_topics.append(self.news_topics[rec_item])
        
        return selected_topics, rec_items

