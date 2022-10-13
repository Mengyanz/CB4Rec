"""Define a NRMS based greedy recommendation policy. """

import math 
import numpy as np 
from collections import defaultdict
import torch 
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import pickle

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
from utils.data_util import read_data, newsample, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news, load_cb_nid2topicindex, SimEvalDataset, SimEvalDataset2, SimTrainDataset

class NeuralGreedy(ContextualBanditLearner):
    def __init__(self, args, device, name='NeuralGreedy'):
        """Use NRMS model. 
        """
        super(NeuralGreedy, self).__init__(args, device, name) 
        self.n_inference = 1
        # self.preinference_mode = self.args.preinference_mode
        self.news_embs = [] # preinferenced news embeddings
        
        # model 
        self.model = NRMS_Model(self.word2vec, news_embedding_dim = args.news_dim).to(self.device)
        
    @torch.no_grad()
    def run_eva(self):
        """For debug: Evaluate model on valid data
        """
        import os
        import sys
        module_path = os.path.abspath(os.path.join('..'))
        if module_path not in sys.path:
            sys.path.append(module_path)
        from CB4Rec.preprocess import eva
        import pickle

        with open(os.path.join(self.args.root_data_dir, self.args.dataset, 'utils', "valid_contexts.pkl"), "rb") as f:
            valid_sam = pickle.load(f)
        val_scores = eva(self.args, self.model, valid_sam, self.nid2index, self.nindex2vec)
        val_auc, val_mrr, val_ndcg, val_ndcg10, ctr = [np.mean(i) for i in list(zip(*val_scores))]
        print(f"Debug: Evaluate model on valid data -- auc: {val_auc:.4f}, mrr: {val_mrr:.4f}, ndcg5: {val_ndcg:.4f}, ndcg10: {val_ndcg10:.4f}, ctr: {ctr:.4f}")
           
    @torch.no_grad()
    def _get_news_embs(self):
        print('Inference news {} times...'.format(self.n_inference))
        news_dataset = NewsDataset(self.nindex2vec) 
        news_dl = DataLoader(news_dataset,batch_size=1024, shuffle=False, num_workers=self.args.num_workers)
        
        self.news_embs = []
        if self.n_inference == 1:
            self.model.eval() # disable dropout
        else:
            self.model.train() # enable dropout, for dropout ucb
        for i in range(self.n_inference): 
            news_vecs = []
            for news in news_dl: 
                news = news.to(self.device)
                news_vec = self.model.text_encoder(news).detach().cpu().numpy()
                news_vecs.append(news_vec)
            self.news_embs.append(np.concatenate(news_vecs))

    @torch.no_grad()
    def _get_user_embs(self, uid, i):
        # get user vect 
        
        h = self.clicked_history[uid]
        h = h + [0] * (self.args.max_his_len - len(h))
        h = self.news_embs[i][np.array(h)]
        h = torch.Tensor(h[None,:,:]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            user_emb = self.model.user_encoder(h).detach().cpu().numpy()
        return user_emb

    def construct_trainable_samples(self):
        """construct trainable samples which will be used in NRMS model training
        from self.h_contexts, self.h_actions, self.h_rewards
        """
        tr_samples = []

        for i, l in enumerate(self.data_buffer):
            poss, negs, his, uid, tsp = l
            if len(poss) > 0 and len(negs) > 0:  
                # negs = newsample(negs, npratio)
                for pos in poss:
                    tr_samples.append([pos, negs, his, uid, tsp])
        return tr_samples

    def train(self): 
        # update learner
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        ft_sam = self.construct_trainable_samples()
        # print('Debug ft_sam: ', ft_sam)
        if len(ft_sam) > 0:
            print('Updating the internal neural model of the bandit!')
            ft_ds = TrainDataset(self.args, ft_sam, self.nid2index, self.nindex2vec)
            ft_dl = DataLoader(ft_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            
            # do one epoch only
            loss = 0
            self.model.train()
            # ft_loader = tqdm(ft_dl)
            for cnt, batch_sample in enumerate(ft_dl):
                candidate_news_index, his_index, label = batch_sample
                candidate_news_index = candidate_news_index.to(self.device)
                his_index = his_index.to(self.device)
                label = label.to(self.device)
                bz_loss, y_hat = self.model(candidate_news_index, his_index, label)

                loss += bz_loss.detach().cpu().numpy()
                optimizer.zero_grad()
                bz_loss.backward()

                optimizer.step()  
            self._get_news_embs() # update news embeddings
            if self.args.reset_buffer:
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
        # cand_news.remove('<unk>')
        if len(self.news_embs) < 1:
            self._get_news_embs() # init news embeddings
        user_vecs = self._get_user_embs(uid, 0) # (b,d)
        scores = self.news_embs[0][cand_news] @ user_vecs.T # (n,b) 
        nid_argmax = np.argsort(scores.squeeze(-1))[::-1][:m].tolist() # (len(uids),)
        rec_itms = [cand_news[n] for n in nid_argmax]
        return rec_itms 

        # if self.preinference_mode:
        #     assert self.n_inference == 1
        #     user_vecs = self._get_user_embs(uid, 0) # (b,d)
        #     scores = self.news_embs[0][cand_news] @ user_vecs.T # (n,b) 
        #     nid_argmax = np.argsort(scores.squeeze(-1))[::-1][:m].tolist() # (len(uids),)
        #     rec_itms = [cand_news[n] for n in nid_argmax]
        #     return rec_itms 
        # else:
        #     batch_size = min(self.args.max_batch_size, len(cand_news))
        #     # get user vect 
        #     h = self.clicked_history[uid]
        #     h = h + [0] * (self.args.max_his_len - len(h))
        #     h = self.nindex2vec[h]
        #     h = torch.Tensor(h[None,:,:])
        #     sed = SimEvalDataset2(self.args, cand_news, self.nindex2vec)
        #     rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers) 

        #     scores = []
        #     for cn in rdl:
        #         score = self.model.forward(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, compute_loss=False)
        #         scores.append(score.detach().cpu().numpy()) 
        #     scores = np.concatenate(scores).squeeze(-1)
        #     # print(scores.shape)   

        #     nid_argmax = np.argsort(scores)[::-1][:m].tolist() # (len(uids),)
        #     rec_itms = [cand_news[n] for n in nid_argmax]
        #     return rec_itms 
    
    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Save and Reset the CB learner to its initial state (do this for each trial/experiment). """
          
        self.model = NRMS_Model(self.word2vec, news_embedding_dim = self.args.news_dim).to(self.device)
        if reload_flag: # and reload_path is not None:
            print('Info: reload cb model from {}'.format(reload_path))
            with open(os.path.join(reload_path, "{}_clicked_history.pkl".format(e)), 'rb') as f:
                self.clicked_history = pickle.load(f) 
            self.data_buffer = np.load(os.path.join(reload_path, "{}_data_buffer.npy".format(e)), allow_pickle=True).tolist()
            state = torch.load(os.path.join(reload_path, '{}_nrms.pkl'.format(e)))
            self.model.load_state_dict(state['state_dict'])
            # self.optimizer.load_state_dict(state['optimizer'])
        else:
            print('Info: reset without reload!')
            self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
            self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  

    def save(self, e=None):
        """Save the CB learner for future reload to continue run more iterations.
        Args
            e: int
                trial
        """
        try:
            model_path = os.path.join(self.args.result_path, 'model', self.args.algo_prefix + '-' + str(self.args.T)) # store final results
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            with open(os.path.join(model_path, "{}_clicked_history.pkl".format(e)), "wb") as f:
                pickle.dump(self.clicked_history, f)
            np.save(os.path.join(model_path, "{}_data_buffer".format(e)), self.data_buffer)

            state = {
                'state_dict': self.model.state_dict(),
                # 'optimizer': self.optimizer.state_dict()
            }
            torch.save(state, os.path.join(model_path, '{}_nrms.pkl'.format(e)))
            print('Info: model saved at {}'.format(model_path))
        except AttributeError:
            print('Warning: no attribute clicked_history find. Skip saving.')
            pass 
            
class Two_NeuralGreedy(NeuralGreedy):
    def __init__(self, args, device, name='2_neuralgreedy'):
        """Use NRMS model. 
        """
        super(Two_NeuralGreedy, self).__init__(args, device, name) 
        self.n_inference = 1
        topic_list, nid2topic = load_cb_topic_news(args, ordered=True) # topic_list: a list of all the topic names, the order of them matters; newsid_to_topic: a dict that maps newsid to topic
        self.nid2topic = nid2topic
        self.nid2topicindex = load_cb_nid2topicindex(args)
        self.topic_order = [i for i in range(len(topic_list))]
        self.index2nid = {v:k for k,v in self.nid2index.items()}

        # model 
        self.topic_model = NRMS_Topic_Model(self.word2vec, split_large_topic=args.split_large_topic, num_categories=self.args.num_topics).to(self.device)
        # print("topic_model text embeddding size: ", self.topic_model.text_encoder.word_embedding.weight.size())
        # print("topic_model topic embedding size: ", self.topic_model.topic_encoder.word_embedding.weight.size())

        # self.cb_topics = list(self.cb_news.keys())
        self.cb_topics = topic_list

        self.topic_budget = len(self.cb_topics) # the score budget for topic exploration
        
        self.gamma = self.args.gamma

    @torch.no_grad()
    def _get_news_embs(self, topic=False):
        print('Inference news {} times...'.format(self.n_inference))
        news_dataset = NewsDataset(self.nindex2vec) 
        news_dl = DataLoader(news_dataset,batch_size=1024, shuffle=False, num_workers=self.args.num_workers)
        
        if not topic:  
            self.news_embs = []
            if self.n_inference == 1:
                self.model.eval() # disable dropout
            else:
                self.model.train() # enable dropout, for dropout ucb
            for i in range(self.n_inference): 
                news_vecs = []
                for news in news_dl: 
                    news = news.to(self.device)
                    news_vec = self.model.text_encoder(news).detach().cpu().numpy()
                    news_vecs.append(news_vec)
                self.news_embs.append(np.concatenate(news_vecs))
        else:
            self.topic_news_embs = []
            if self.n_inference == 1:
                self.topic_model.eval() # disable dropout
            else:
                self.topic_model.train() # enable dropout, for dropout ucb
            for i in range(self.n_inference): 
                news_vecs = []
                for news in news_dl: 
                    news = news.to(self.device)
                    news_vec = self.topic_model.text_encoder(news).detach().cpu().numpy()
                    news_vecs.append(news_vec)
                self.topic_news_embs.append(np.concatenate(news_vecs))

    @torch.no_grad()
    def _get_topic_user_embs(self, uid, i):
        if self.n_inference == 1:
            self.topic_model.eval() # disable dropout
        else:
            self.topic_model.train() # enable dropout, for dropout ucb
        h = self.clicked_history[uid]
        h = h + [0] * (self.args.max_his_len - len(h))
        # h = torch.LongTensor(self.nindex2vec[h]).to(self.device)
        # h = self.topic_model.text_encoder(h).unsqueeze(0)
        h = self.topic_news_embs[i][np.array(h)]
        h = torch.Tensor(h[None,:,:]).to(self.device)
        user_vector = self.topic_model.dimmension_reduction(self.topic_model.user_encoder(h)).squeeze(0) # 1 x reduction
        return user_vector

    def train(self, mode='item'): 
        # update learner
        if mode == 'item':
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
            ft_sam = self.construct_trainable_samples()
            if len(ft_sam) > 0:
                print('Updating the internal item neural model of the bandit!')
                ft_ds = TrainDataset(self.args, ft_sam, self.nid2index, self.nindex2vec)
                ft_dl = DataLoader(ft_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
                # ft_loader = tqdm(ft_dl)
                
                # do one epoch only
                loss = 0
                self.model.train()
                for cnt, batch_sample in enumerate(ft_dl):
                    candidate_news_index, his_index, label = batch_sample
                    candidate_news_index = candidate_news_index.to(self.device)
                    his_index = his_index.to(self.device)
                    label = label.to(self.device)
                    bz_loss, y_hat = self.model(candidate_news_index, his_index, label)

                    loss += bz_loss.detach().cpu().numpy()
                    optimizer.zero_grad()
                    bz_loss.backward()

                    optimizer.step()  
                self._get_news_embs() # update news embeddings#
                # REVIEW: assume topics is updated more frequently than items
                if self.args.reset_buffer:
                    self.data_buffer = [] # reset data buffer
            else:
                print('Skip update cb item learner due to lack valid samples!')
            
        elif mode == 'topic':
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
            ft_sam = self.construct_trainable_samples()
            if len(ft_sam) > 0:
                print('Updating the internal topic neural model of the bandit!')
                ft_ds = TrainDataset(self.args, ft_sam, self.nid2index, self.nindex2vec, self.nid2topicindex)
                ft_dl = DataLoader(ft_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
                # ft_loader = tqdm(ft_dl)
                
                # do one epoch only
                loss = 0
                self.topic_model.train()
                for cnt, batch_sample in enumerate(ft_dl):
                    candidate_news_index, his_index, label = batch_sample
                    candidate_news_index = candidate_news_index.to(self.device)
                    his_index = his_index.to(self.device)
                    label = label.to(self.device)
                    bz_loss, y_hat = self.topic_model(candidate_news_index, his_index, label)

                    loss += bz_loss.detach().cpu().numpy()
                    optimizer.zero_grad()
                    bz_loss.backward()

                    optimizer.step()  
                self._get_news_embs(topic=True) # update topic embeddings
                # REVIEW: not reset data buffer 
                # self.data_buffer = [] # reset data buffer
            else:
                print('Skip update cb topic learner due to lack valid samples!')

    def update(self, topics, items, rewards, mode = 'topic', uid = None):
        """Update its internal model. 

        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its numerical index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item` 
        """
        # Update the user_encoder(topic),news_encoder(topic),topic_encoder using `self.clicked_history`
        print('size(data_buffer): {}'.format(len(self.data_buffer)))      
        self.train(mode=mode)

    @torch.no_grad()
    def topic_rec(self, uid, m=1):
        """
        Args:
            uid: str, a user id 
            m: int, number of items to rec 
        Return: 
            list, containing m element, where each element is a list of cand news index inside a topic (topic can be newly formed if we dynamically form topics)
        """
        if len(self.news_embs) < 1:
            self._get_news_embs(topic=True) # init news embeddings
        
        user_vector = self._get_topic_user_embs(uid, 0) # reduction_dim
        topic_embeddings = self.topic_model.get_topic_embeddings_byindex(self.topic_order) # get all active topic scores, num x reduction_dim
        scores = (topic_embeddings @ user_vector.unsqueeze(-1)).squeeze(-1).cpu().numpy() # num_topic
        sorted_topic_indexs = np.argsort(scores)[::-1].tolist() 
        # rec_topic = [self.active_topics[n] for n in nid_argmax]
        recs = self.topic_cand_news_prep(sorted_topic_indexs,m)
        return recs
    
    def topic_cand_news_prep(self, sorted_topic_indexs, m=1):
        recs = [] # each element is a list of cand news index inside a topic (topic can be newly formed if we dynamically form topics)
        recs_topic = [] # each element is a list of topic names
        recs_size = [] # each element is the size of the new topic
        rank = 0

        # for i in range(m):
        #     cand_news = []
        #     topics = []
        #     while len(cand_news) < self.args.min_item_size:
        #         topic_idx = sorted_topic_indexs[rank]
        #         topic = self.cb_topics[topic_idx]
        #         cand_news.extend([self.nid2index[n] for n in self.cb_news[topic]])
        #         rank += 1
        #         topics.append(topic)
        #         if not self.args.dynamic_aggregate_topic:
        #             break
        #     recs.append(cand_news)
        #     recs_size.append(len(cand_news))
        #     recs_topic.append(topics)

        cand_news = []
        topics = []
        left_news_indexes = list(range(self.args.num_all_news))
        for i in range(m):
            topic_idx = sorted_topic_indexs[rank]
            topic = self.cb_topics[topic_idx]
            news_index_topic = [self.nid2index[n] for n in self.cb_news[topic]]
            left_news_indexes = list(set(left_news_indexes) - set(news_index_topic))
            cand_news.append(news_index_topic)
            rank += 1
            topics.append([topic])
        if self.args.dynamic_aggregate_topic:
            unfinished_topics = list(range(m))
            while len(unfinished_topics) > 0:
                for i in unfinished_topics:
                    num_news_to_add = self.args.min_item_size - len(cand_news[i])
                    if num_news_to_add > 0 and rank < len(self.cb_topics):
                        topic_idx = sorted_topic_indexs[rank]
                        topic = self.cb_topics[topic_idx]
                        news_index_topic = [self.nid2index[n] for n in self.cb_news[topic]]
                        if len(news_index_topic) <= num_news_to_add:
                            cand_news[i].extend(news_index_topic)
                        else:
                            sample_news_index = np.random.choice(news_index_topic, size= num_news_to_add, replace=False)
                            cand_news[i].extend(sample_news_index)
                        rank += 1
                        topics[i].extend([topic])
                    else:
                        unfinished_topics.remove(i)
        else:
            for i in range(len(topics)):
                num_news_to_add = self.args.min_item_size - len(cand_news[i])
                if num_news_to_add > 0:
                    sample_news_index = np.random.choice(left_news_indexes, size= num_news_to_add, replace=False)
                    cand_news[i].extend(sample_news_index)
                    left_news_indexes = list(set(left_news_indexes) - set(sample_news_index))

                        
        recs = cand_news
        recs_topic = topics
        for i in range(m):
            recs_size.append(len(recs[i]))
            
        # rec_topic = [self.cb_topics[n] for n in nid_argmax]
        # sort topic by topic size, from small to large
        # topic_size = [len(self.cb_news[n]) for n in rec_topic]
        recs = [recs[i] for i in np.argsort(recs_size)]
        recs_topic = [recs_topic[i] for i in np.argsort(recs_size)]
        for i in range(m):
            print('Debug rec topic {}, total size {}'.format(recs_topic[i], len(recs[i])))
        return recs
        

    def sample_actions(self, uid, cand_news = None):
        """Choose an action given a context. 
        
        Args:
            uids: a list of str uIDs (user ids). 
            cand_news: list of candidate news indexes 
        Return: 
            topics: (len(uids), `rec_batch_size`)
            items: (len(uids), `rec_batch_size`) 
        """
        rec_news_indexs = self.topic_rec(uid, m=self.rec_batch_size)
        left_budget = self.per_rec_score_budget * self.rec_batch_size - len(self.cb_topics)
        left_to_rec = self.rec_batch_size
        rec_items = []

        for news_indexs in rec_news_indexs:
            left_budget_ave = int(left_budget/left_to_rec)
            allocate_budget = min(len(news_indexs), left_budget_ave)
            print('Randomly sample {} candidates news out of candidate news ({})'.format(allocate_budget, len(news_indexs)))
            # news_indexs = [self.nid2index[n] for n in self.cb_news[topic]]
            cand_news = np.random.choice(news_indexs, size=allocate_budget, replace=False)
            rec_item = self.item_rec(uid, cand_news)
            rec_items.append(rec_item[0].item()) # Convert numpy int64 to python int

            left_budget -= allocate_budget
            left_to_rec -= 1 
            rec_topics = []
            for n in rec_items:
                try:
                    rec_topics.append(self.nid2topicindex[self.index2nid[n]])
                except:
                    print('n: ', n)
                    print('self.index2nid[n]: ', self.index2nid[n])
                    print('self.nid2topicindex[self.index2nid[n]]: ', self.nid2topicindex[self.index2nid[n]])

        return rec_topics, rec_items
    
    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Save and Reset the CB learner to its initial state (do this for each trial/experiment). """
          
        self.model = NRMS_Model(self.word2vec, news_embedding_dim = self.args.news_dim).to(self.device)
        self.topic_model = NRMS_Topic_Model(self.word2vec, split_large_topic=self.args.split_large_topic, num_categories=self.args.num_topics).to(self.device)
        if reload_flag: # and reload_path is not None:
            print('Info: reload cb model from {}'.format(reload_path))
            with open(os.path.join(reload_path, "{}_clicked_history.pkl".format(e)), 'rb') as f:
                self.clicked_history = pickle.load(f) 
            self.data_buffer = np.load(os.path.join(reload_path, "{}_data_buffer.npy".format(e)), allow_pickle=True).tolist()
            state = torch.load(os.path.join(reload_path, '{}_nrms.pkl'.format(e)))
            self.model.load_state_dict(state['state_dict'])
            self.topic_model.load_state_dict(state['topic_state_dict'])
        else:
            print('Info: reset without reload!')
            self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
            self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected 
            
    def save(self, e=None):
        """Save the CB learner for future reload to continue run more iterations.
        Args
            e: int
                trial
        """
        try:
            model_path = os.path.join(self.args.result_path, 'model', self.args.algo_prefix + '-' + str(self.args.T)) # store final results
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            with open(os.path.join(model_path, "{}_clicked_history.pkl".format(e)), "wb") as f:
                pickle.dump(self.clicked_history, f)
            np.save(os.path.join(model_path, "{}_data_buffer".format(e)), self.data_buffer)

            state = {
                'state_dict': self.model.state_dict(),
                'topic_state_dict': self.topic_model.state_dict(),
                # 'optimizer': self.optimizer.state_dict()
            }
            torch.save(state, os.path.join(model_path, '{}_nrms.pkl'.format(e)))
            print('Info: model saved at {}'.format(model_path))
        except AttributeError:
            print('Warning: no attribute clicked_history find. Skip saving.')
            pass 
            
    
    # def item_rec(self, uid, cand_news): 
    #     """
    #     Args:
    #         uid: str, a user id 
    #         cand_news: a list of int (not nIDs but their index version from `nid2index`) 
    #     Return: 
    #         item: int 
    #     """

    #     score_budget = self.per_rec_score_budget - int(self.topic_budget/self.rec_batch_size)
    #     if len(cand_news)>score_budget:
    #         print('Randomly sample {} candidates news out of candidate news ({})'.format(score_budget, len(cand_news)))
    #         cand_news = np.random.choice(cand_news, size=score_budget, replace=False).tolist()
   
    #     all_scores = []
    #     for i in range(self.n_inference): 
    #         user_vecs = self._get_user_embs(uid, i) # (b,d)
    #         scores = self.news_embs[i][cand_news] @ user_vecs.T # (n,b) 
    #         all_scores.append(scores) 
    #     all_scores = np.array(all_scores).squeeze(-1)  # (n_inference,n,b)
    #     mu = np.mean(all_scores, axis=0) 
    #     std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
    #     ucb = mu + self.gamma * std # (n,) 
    #     nid_argmax = np.argmax(ucb).tolist()
    #     return cand_news[nid_argmax]

    # @torch.no_grad()
    # def topic_rec(self, uid):
    #     """
    #     Args:
    #         uid: str, a user id 
    #     Return: 
    #         topic: str 
    #     """

    #     self.topic_model.train()
    #     all_scores = []
    #     for i in range(self.args.n_inference):
    #         user_vector = self._get_topic_user_embs(uid, i) # reduction_dim
    #         topic_embeddings = self.topic_model.get_topic_embeddings_byindex(self.active_topics_order) # get all active topic scores, num x reduction_dim
    #         score = (topic_embeddings @ user_vector.unsqueeze(-1)).squeeze(-1).cpu().numpy() # num_topic
    #         all_scores.append(score)

    #     all_scores = np.array(all_scores) # n_inference, num_active_topic
    #     mu = np.mean(all_scores, axis=0) 
    #     std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference)
    #     # print('Debug topic std: ', std) 
    #     ucb = mu + self.gamma * std  # num_topic
    #     # for topic in self.active_topics:
    #     #     s = np.random.beta(a= self.alphas[topic], b= self.betas[topic])
    #     #     ss.append(s)
    #     rec_topic = self.active_topics[np.argmax(ucb)]
    #     return rec_topic

    # def sample_actions(self, uid): 
    #     """Choose an action given a context. 
    #     Args:
    #         uid: str, user id
    #     Return: 
    #         topics: (`rec_batch_size`)
    #         items: (`rec_batch_size`) @TODO: what if one topic has less than `rec_batch_size` numbers of items? 
    #     """
    #     rec_topics = []
    #     rec_items = []
    #     if len(self.news_embs) < 1:
    #         self._get_news_embs() # init news embeddings
    #     if len(self.topic_news_embs) < 1:
    #         self._get_news_embs(topic=True)
    #     self.active_topics = self.cb_topics.copy()
    #     self.active_topics_order = self.topic_order.copy()
    #     while len(rec_items) < self.rec_batch_size:
    #         cand_news = []
    #         while len(cand_news) < self.args.min_item_size:
    #             rec_topic = self.topic_rec(uid)
    #             rec_topics.append(rec_topic)
    #             rec_topic_pos = self.active_topics.index(rec_topic)
    #             self.active_topics.remove(rec_topic)
    #             del self.active_topics_order[rec_topic_pos]

    #             cand_news.extend([self.nid2index[n] for n in self.cb_news[rec_topic]])
    #             if not self.args.dynamic_aggregate_topic:
    #                 print('Debug dynamic_aggregate_topic', self.args.dynamic_aggregate_topic)
    #                 break
    #             # DEBUG
    #         print('DEBUG:', rec_topic, len(cand_news))
    #         rec_item = self.item_rec(uid, cand_news)
    #         rec_items.append(rec_item)
        
    #     return rec_topics, rec_items
            
        