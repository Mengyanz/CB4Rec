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
from algorithms.neural_greedy import SingleStageNeuralGreedy
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news,load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class SingleStageNeuralUCB(SingleStageNeuralGreedy):
    def __init__(self,device, args, name='SingleStageNeuralUCB'):
        """Use NRMS model. 
        """      
        super(SingleStageNeuralUCB, self).__init__(device, args, name)
        self.n_inference = self.args.n_inference 
        self.gamma = self.args.gamma
        # self.cb_indexs = self._get_cb_news_index([item for sublist in list(self.cb_news.values()) for item in sublist])

        # # pre-generate news embeddings
        # self.news_vecss = []
        # for i in range(self.n_inference): 
        #     news_vecs = self._get_news_vecs() # (n,d)
        #     self.news_vecss.append(news_vecs) 

    # def _get_cb_news_index(self, cb_news):
    #     """Generate cb news vecs by inferencing model on cb news

    #     Args
    #         cb_news: list of cb news samples

    #     Return
    #         cb_indexs: list of indexs corresponding to the input cb_news
    #     """
    #     print('#cb news: ', len(cb_news))
    #     cb_indexs = []
    #     for l in cb_news:
    #         nid = l.strip('\n').split("\t")[0]
    #         cb_indexs.append(self.nid2index[nid])
    #     return np.array(cb_indexs)     


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
            all_scores = []
                
            for i in range(self.n_inference): 
                user_vecs = self._get_user_embs(uid, i) # (b,d)
                scores = self.news_embs[i][cand_news] @ user_vecs.T # (n,b) 
                all_scores.append(scores) 
            
            all_scores = np.array(all_scores).squeeze(-1)  # (n_inference,n,b)
            mu = np.mean(all_scores, axis=0) 
            std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
            ucb = mu + self.gamma * std # (n,) 
            nid_argmax = np.argsort(ucb, axis = 0)[::-1][:m].tolist() # (len(uids),)
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

            all_scores = []
            for i in range(self.n_inference): # @TODO: accelerate
                scores = []
                for cn in rdl:
                    score = self.model.forward(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, compute_loss=False)
                    scores.append(score.detach().cpu().numpy()) 
                scores = np.concatenate(scores)
                all_scores.append(scores) 

            all_scores = np.array(all_scores).squeeze(-1) # (n_inference,len(cand_news))
            mu = np.mean(all_scores, axis=0) 
            std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
            ucb = mu + self.gamma * std # (n,b) 
        
            nid_argmax = np.argsort(ucb)[::-1][:m].tolist() # (len(uids),)
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
        # all_scores = []
        # self.model.eval()
        # for i in range(self.n_inference): # @TODO: accelerate
        #     user_vecs = self._get_user_embs(self.news_vecss[i], user_samples) # (b,d)
        #     scores = self.news_vecss[i][self.cb_indexs] @ user_vecs.T # (n,b) 
        #     all_scores.append(scores) 
        
        # all_scores = np.array(all_scores) # (n_inference,n,b)
        # mu = np.mean(all_scores, axis=0) 
        # std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
        # ucb = mu + std # (n,b) 
        # sorted_ids = np.argsort(ucb, axis=0)[-self.rec_batch_size:,:] 
        # return self.cb_indexs[sorted_ids], np.empty(0)
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings

        cand_news = [self.nid2index[n] for n in self.cb_news]
        rec_items = self.item_rec(uids, cand_news, self.rec_batch_size)

        return np.empty(0), rec_items


class TwoStageNeuralUCB(SingleStageNeuralUCB):
    def __init__(self,device, args, name='ThompsonSampling_NeuralUCB'):
        """Two stage exploration. Use NRMS model. 
        """
        super(TwoStageNeuralUCB, self).__init__(device, args, name)
        
        topic_news = load_cb_topic_news(args) # dict, key: subvert; value: list nIDs 
        cb_news = defaultdict(list)
        for k,v in topic_news.items():
            cb_news[k] = [l.strip('\n').split("\t")[0] for l in v] # get nIDs 
        self.cb_news = cb_news 
        self.cb_topics = list(self.cb_news.keys())
        self.uniform_init = self.args.uniform_init 

        self.alphas = {}
        self.betas = {}

    def set_clicked_history(self, init_clicked_history):
        """
        Args:
            init_click_history: list of init clicked history nindexs
        """
        self.clicked_history = init_clicked_history
        if self.uniform_init:
            for topic in self.cb_topics:
                self.alphas[topic] = 1
                self.betas[topic] = 1
        else:
            print('Debug non uniform init for ts!')
            clicked_nindexs = np.concatenate(list(init_clicked_history.values()))
            ave_clicks = len(clicked_nindexs)/len(self.cb_topics)
            print('Debug n_clicks all: ', len(clicked_nindexs))
            print('Debug ave_clicks over topics: ', ave_clicks)
            for topic in self.cb_topics:
                topic_nindexs = [self.nid2index[n] for n in self.cb_news[topic]]
                self.alphas[topic] = len([nindex for nindex in clicked_nindexs if nindex in set(topic_nindexs)]) # n_clicks of topic in clicked histories
                # print('Debug topic {} with alpha init as {}'.format(topic, self.alphas[topic]))
                self.betas[topic] = ave_clicks

    def topic_rec(self):
        """    
        Return
            rec_topic: one recommended topic
        """
        # TODO: topic selection also needs score budget allocation
        ss =[] 
        for topic in self.active_topics:
            s = np.random.beta(a= self.alphas[topic], b= self.betas[topic])
            ss.append(s)
        rec_topic = self.active_topics[np.argmax(ss)]
        return rec_topic

    # def item_rec(self, user_samples, cand_news):
    #     all_scores = []
    #     self.model.eval()
    #     cb_indexs = self._get_cb_news_index(cand_news)
    #     for i in range(self.n_inference): 
    #         user_vecs = self._get_user_embs(self.news_vecss[i], user_samples) # (b,d)
    #         scores = self.news_vecss[i][cb_indexs] @ user_vecs.T # (n,b) 
    #         all_scores.append(scores) 
        
    #     all_scores = np.array(all_scores) # (n_inference,n,b)
    #     mu = np.mean(all_scores, axis=0) 
    #     std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
    #     ucb = mu + std # (n,b) 
    #     sorted_ids = np.argsort(ucb, axis=0)[-1,:] 
    #     return cb_indexs[sorted_ids]

    def update(self, topics, items, rewards, mode = 'topic', uid = None):
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
                print("attention::", rewards[i])
                assert rewards[i] in {0,1}
                self.alphas[topic] += rewards[i]
                self.betas[topic] += 1 - rewards[i]

        # Update the user_encoder and news_encoder using `self.clicked_history`
        if mode == 'item': 
            self.train() 
            self._get_news_embs()

    def sample_actions(self, uid): 
        """Choose an action given a context. 
        Args:
            uid: str, user id

        Return: 
            topics: (`rec_batch_size`)
            items: (`rec_batch_size`) @TODO: what if one topic has less than `rec_batch_size` numbers of items? 
        """
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings

        rec_topics = []
        rec_items = []
        self.active_topics = self.cb_topics.copy()
        while len(rec_items) < self.rec_batch_size:
            # todo: uncomment these to add dynamic_aggregate_topic 
            cand_news = []
            while len(cand_news) < self.args.min_item_size:
                rec_topic = self.topic_rec()
                rec_topics.append(rec_topic)
                self.active_topics.remove(rec_topic)
                cand_news.extend([self.nid2index[n] for n in self.cb_news[rec_topic]])
                if not self.args.dynamic_aggregate_topic:
                    break
            # rec_topic = self.topic_rec()
            # rec_topics.append(rec_topic)
            # self.active_topics.remove(rec_topic)

            # cand_news = [self.nid2index[n] for n in self.cb_news[rec_topic]]
            # DEBUG
            print('DEBUG:', rec_topic, len(cand_news))
            rec_item = self.item_rec(uid, cand_news,m=1)[0]
            rec_items.append(rec_item)
        
        return rec_topics, rec_items
      
class DummyTwoStageNeuralUCB(ContextualBanditLearner): #@Thanh: for the sake of testing my pipeline only 
    def __init__(self,device, args, name='TwoStageNeuralUCB'):
        """Two stage exploration. Use NRMS model. 
        """
        super(DummyTwoStageNeuralUCB, self).__init__(args, name)
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
        if self.pretrained_mode == 'pretrained':
            self.model.load_state_dict(torch.load(args.learner_path)) 
 
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
        rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=4) 

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
    
    
class TwoStageNeuralUCB_zhenyu(SingleStageNeuralUCB):  #@ZhenyuHe: for the sake of testing my pipeline only 
    def __init__(self,device, args, name='NeuralUCB_NeuralUCB'):
        """Two stage exploration. Use NRMS model. 
            Args:
                rec_batch_size: int, recommendation size. 
                n_inference: int, number of Monte Carlo samples of prediction. 
                pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 
        """
        super(TwoStageNeuralUCB_zhenyu, self).__init__(device, args, name)
        self.args = args
        self.n_inference = args.n_inference
        self.name = name 
        self.device = device 

        # preprocessed data 
        self.nid2index, word2vec, self.nindex2vec = load_word2vec(args)
        topic_news = load_cb_topic_news(args) # dict, key: subvert; value: list nIDs 
        topic_list, nid2topic = load_cb_topic_news(args, ordered=True) # topic_list: a list of all the topic names, the order of them matters; newsid_to_topic: a dict that maps newsid to topic
        self.nid2topic = nid2topic
        self.nid2topicindex = load_cb_nid2topicindex(args)
        self.topic_order = [i for i in range(len(topic_list))]
        cb_news = defaultdict(list)
        for k,v in topic_news.items():
            cb_news[k] = [l.strip('\n').split("\t")[0] for l in v] # get nIDs 
        self.cb_news = cb_news 

        # model 
        self.topic_model = NRMS_Topic_Model(word2vec, split_large_topic=args.split_large_topic).to(self.device)
        if self.pretrained_mode == 'pretrained':
            self.topic_model.load_state_dict(torch.load(args.learner_path)) 
        print("topic_model text embeddding size: ", self.topic_model.text_encoder.word_embedding.weight.size())
        print("topic_model topic embedding size: ", self.topic_model.topic_encoder.word_embedding.weight.size())
            
        # self.cb_topics = list(self.cb_news.keys())
        self.cb_topics = topic_list

        self.topic_news_embs = []

        self.alphas = {}
        self.betas = {}

        for topic in self.cb_topics:
            self.alphas[topic] = 1
            
    @torch.no_grad()
    def _get_news_embs(self, topic=False):
        print('Inference news {} times...'.format(self.n_inference))
        news_dataset = NewsDataset(self.nindex2vec) 
        news_dl = DataLoader(news_dataset,batch_size=1024, shuffle=False, num_workers=2)
        news_vecs = []
        if not topic:
            self.news_embs = []
            for i in range(self.n_inference): 
                for news in news_dl: # @TODO: avoid for loop
                    news = news.to(self.device)
                    news_vec = self.model.text_encoder(news).detach().cpu().numpy()
                    news_vecs.append(news_vec)
                self.news_embs.append(np.concatenate(news_vecs))
        else:
            self.topic_news_embs = []
            for i in range(self.n_inference): 
                for news in news_dl: # @TODO: avoid for loop
                    news = news.to(self.device)
                    news_vec = self.topic_model.text_encoder(news).detach().cpu().numpy()
                    news_vecs.append(news_vec)
                self.topic_news_embs.append(np.concatenate(news_vecs))

    @torch.no_grad()
    def _get_topic_user_embs(self, uid, i):
        h = self.clicked_history[uid]
        h = h + [0] * (self.args.max_his_len - len(h))
        # h = torch.LongTensor(self.nindex2vec[h]).to(self.device)
        # h = self.topic_model.text_encoder(h).unsqueeze(0)
        h = self.topic_news_embs[i][np.array(h)]
        h = torch.Tensor(h[None,:,:]).to(self.device)
        user_vector = self.topic_model.dimmension_reduction(self.topic_model.user_encoder(h)).squeeze(0) # 1 x reduction
        
        return user_vector

    @torch.no_grad()
    def topic_rec(self, uid):
        """
        Args:
            uid: str, a user id 
        Return: 
            topic: str 
        """

        self.topic_model.train()
        all_scores = []
        for i in range(self.args.n_inference):
            user_vector = self._get_topic_user_embs(uid, i) # reduction_dim
            topic_embeddings = self.topic_model.get_topic_embeddings_byindex(self.active_topics_order) # get all active topic scores, num x reduction_dim
            score = (topic_embeddings @ user_vector.unsqueeze(-1)).squeeze(-1).cpu().numpy() # num_topic
            all_scores.append(score)

        all_scores = np.array(all_scores) # n_inference, num_active_topic
        mu = np.mean(all_scores, axis=0) 
        std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
        ucb = mu + std  # num_topic
        # for topic in self.active_topics:
        #     s = np.random.beta(a= self.alphas[topic], b= self.betas[topic])
        #     ss.append(s)
        rec_topic = self.active_topics[np.argmax(ucb)]
        return rec_topic
    

    def item_rec(self, uid, cand_news): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 

        Return: 
            item: int 
        """

        if self.preinference_mode:
            all_scores = []
            for i in range(self.n_inference): 
                user_vecs = self._get_user_embs(uid, i) # (b,d)
                scores = self.news_embs[i][cand_news] @ user_vecs.T # (n,b) 
                all_scores.append(scores) 
            all_scores = np.array(all_scores).squeeze(-1)  # (n_inference,n,b)
            mu = np.mean(all_scores, axis=0) 
            std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
            ucb = mu + self.gamma * std # (n,) 
            nid_argmax = np.argmax(ucb).tolist()
            return cand_news[nid_argmax]


        else:
            batch_size = min(self.args.max_batch_size, len(cand_news))
            # get user vect 
        
            h = self.clicked_history[uid]
            h = h + [0] * (self.args.max_his_len - len(h))
            h = self.nindex2vec[h]

            h = torch.Tensor(h[None,:,:])
            sed = SimEvalDataset2(self.args, cand_news, self.nindex2vec)
            #TODO: Use Dataset is clean and good when len(uids) is large. When len(uids) is small, is it faster to not use Dataset?
            rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=4) 

            all_scores = []
            for _ in range(self.args.n_inference):
                scores = []
                for cn in rdl:
                    score = self.model.forward(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, compute_loss=False)
                    scores.append(score.detach().cpu().numpy()) 
                scores = np.concatenate(scores) 
                all_scores.append(scores) 

            all_scores = np.array(all_scores).squeeze(-1) # (n_inference,len(cand_news))
            mu = np.mean(all_scores, axis=0) 
            std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
            ucb = mu + self.gamma * std # (n,b) 
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

    def train(self, mode='item'):
        
        if mode == 'item':
            # update learner
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
            ft_sam = self.construct_trainable_samples()
            if len(ft_sam) > 0:
                print('Updating the internal item model of the bandit!')
                ft_ds = TrainDataset(self.args, ft_sam, self.nid2index, self.nindex2vec)
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
        
        elif mode == 'topic':
            # update learner
            optimizer = optim.Adam(self.topic_model.parameters(), lr=self.args.lr)
            ft_sam = self.construct_trainable_samples()
            if len(ft_sam) > 0:
                print('Updating the internal topic model of the bandit!')
                ft_ds = TrainDataset(self.args, ft_sam, self.nid2index, self.nindex2vec, self.nid2topicindex)
                ft_dl = DataLoader(ft_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
                
                # do one epoch only
                loss = 0
                self.topic_model.train()
                ft_loader = tqdm(ft_dl)
                for cnt, batch_sample in enumerate(ft_loader):
                    candidate_news_index, his_index, label = batch_sample
                    sample_num = candidate_news_index.shape[0]
                    candidate_news_index = candidate_news_index.to(self.device)
                    his_index = his_index.to(self.device)
                    label = label.to(self.device)
                    bz_loss, y_hat = self.topic_model(candidate_news_index, his_index, label)

                    loss += bz_loss.detach().cpu().numpy()
                    optimizer.zero_grad()
                    bz_loss.backward()

                    optimizer.step()  
            else:
                print('Skip update cb topic learner due to lack valid samples!')

    def update(self, topics, items, rewards, mode = 'topic', uid = None):
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
        # Update the user_encoder(topic),news_encoder(topic),topic_encoder using `self.clicked_history`
        if mode == 'topic': 
            self.train(mode='topic')
            self._get_news_embs(topic=True)

        # Update the user_encoder and news_encoder using `self.clicked_history`
        if mode == 'item': 
            self.train() 
            self._get_news_embs()

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
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings
        if len(self.topic_news_embs) < 1:
            self._get_news_embs(topic=True)
        self.active_topics = self.cb_topics.copy()
        self.active_topics_order = self.topic_order.copy()
        while len(rec_items) < self.rec_batch_size:
            cand_news = []
            while len(cand_news) < self.args.min_item_size:
                rec_topic = self.topic_rec(uid)
                rec_topics.append(rec_topic)
                rec_topic_pos = self.active_topics.index(rec_topic)
                self.active_topics.remove(rec_topic)
                del self.active_topics_order[rec_topic_pos]

                cand_news.extend([self.nid2index[n] for n in self.cb_news[rec_topic]])
                if not self.args.dynamic_aggregate_topic:
                    break
                # DEBUG
            print('DEBUG:', rec_topic, len(cand_news))
            rec_item = self.item_rec(uid, cand_news)
            rec_items.append(rec_item)
        
        return rec_topics, rec_items
    
    
