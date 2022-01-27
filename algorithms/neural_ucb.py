"""Define a simple UCB. """

import math 
import numpy as np 
import torch 
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.nrms_model import NRMS_Model
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news

class SingleStageNeuralUCB(ContextualBanditLearner):
    def __init__(self,device, args, rec_batch_size = 1, n_inference=10, pretrained_mode=True, name='SingleStageNeuralUCB'):
        """Use NRMS model. 
            Args:
                rec_batch_size: int, recommendation size. 
                n_inference: int, number of Monte Carlo samples of prediction. 
                pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 

        """
        self.n_inference = n_inference 
        super(SingleStageNeuralUCB, self).__init__(args, rec_batch_size)

        self.pretrained_mode = pretrained_mode 
        self.name = name 
        self.device = device 
        self.args = args

        # preprocessed data 
        # self.nid2index, _, self.news_index, embedding_matrix, self.cb_users, self.cb_news = read_data(args, mode='cb') 
        self.nid2index, embedding_matrix, self.news_index = load_word2vec(args)
        self.cb_news = load_cb_topic_news(args)
        self.cb_indexs = self._get_cb_news_index([item for sublist in list(self.cb_news.values()) for item in sublist])

        # model 
        self.model = NRMS_Model(embedding_matrix).to(self.device)
        if self.pretrained_mode == 'pretrained':
            self.model.load_state_dict(torch.load(args.learner_path)) 
        # self.news_vecss, self.cb_vecss, self.cb_indexs = self.inference_cb_news()

        # pre-generate news embeddings
        self.news_vecss = []
        for i in range(self.n_inference): 
            news_vecs = self._get_news_vecs() # (n,d)
            self.news_vecss.append(news_vecs) 

        # internal buffer for update
        self.h_contexts = []
        self.h_actions = []
        self.h_rewards = []

    def _get_cb_news_index(self, cb_news):
        """Generate cb news vecs by inferencing model on cb news

        Args
            cb_news: list of cb news samples

        Return
            cb_indexs: list of indexs corresponding to the input cb_news
        """
        print('#cb news: ', len(cb_news))
        cb_indexs = []
        for l in cb_news:
            nid = l.strip('\n').split("\t")[0]
            cb_indexs.append(self.nid2index[nid])
        return np.array(cb_indexs)


    def construct_trainable_samples(self):
        """construct trainable samples which will be used in NRMS model training
        from self.h_contexts, self.h_actions, self.h_rewards
        """
        tr_samples = []
        for i, l in enumerate(self.h_contexts):
            _, _, his, uid, tsp = l
            poss = []
            negs = []
            for j, reward in enumerate(self.h_rewards[i]):
                if reward == 1:
                    poss.append(self.h_actions[i][j])
                elif reward == 0:
                    negs.append(self.h_actions[i][j])
                else:
                    raise Exception("invalid reward")

            if len(poss) > 0 and len(negs) > 0:  
                if len(negs) == 0:
                    pass
                    # TODO: sample negative samples
                for pos in poss:
                    tr_samples.append([pos, negs, his, uid, tsp])
        return tr_samples

    def update_learner(self, context, actions, rewards):
        
        if len(self.h_contexts) < self.args.update_learn_size:
            # store samples into buffer
            self.h_contexts.append(context)
            self.h_actions.append(actions)
            self.h_rewards.append(rewards)
        else:
            # update learner
            print('Updating the internal model of the bandit!')
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
            ft_sam = self.construct_trainable_samples()
            ft_ds = TrainDataset(self.args, ft_sam, self.nid2index, self.news_index)
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

                # if cnt % 10 == 0:
                #     ft_loader.set_description(f"[{cnt}]steps loss: {loss / (cnt+1):.4f} ")
                #     ft_loader.refresh() 
            
            # clear buffer
            self.h_contexts = []
            self.h_actions = []
            self.h_rewards = []

    def update(self, context, topics, actions, rewards):
        """Update its internal model. 

        Args:
            context: a user sample
            topics: dummy 
            actions: list of actions; len: rec_batch_size
            rewards: list of rewards; len: rec_batch_size
        """
        self.update_learner(context, actions, rewards)
        

    def _get_news_vecs(self):
        news_dataset = NewsDataset(self.news_index) 
        news_dl = DataLoader(news_dataset,batch_size=1024, shuffle=False, num_workers=2)
        news_vecs = []
        for news in news_dl: # @TODO: avoid for loop
            news = news.to(self.device)
            news_vec = self.model.text_encoder(news).detach().cpu().numpy()
            news_vecs.append(news_vec)

        return np.concatenate(news_vecs) # (130381, 400)

    def _get_user_vecs(self, news_vecs, user_samples): 
        """Transform user_samples into representation vectors. 

        Args:
            user_samples: a list of (poss, negs, his, uid, tsp) 

        Return: 
            user_vecs: [None, dim]
        """
        user_dataset = UserDataset(self.args, user_samples, news_vecs, self.nid2index)
        user_vecs = []
        user_dl = DataLoader(user_dataset, batch_size=min(1024, len(user_samples)), shuffle=False, num_workers=2)

        for his_tsp in user_dl:
            his, tsp = his_tsp
            his = his.to(self.device)
            user_vec = self.model.user_encoder(his).detach().cpu().numpy()
            user_vecs.append(user_vec)
            # print(tsp)
        return np.concatenate(user_vecs)

    def sample_actions(self, user_samples): 
        all_scores = []
        self.model.eval()
        for i in range(self.n_inference): # @TODO: accelerate
            user_vecs = self._get_user_vecs(self.news_vecss[i], user_samples) # (b,d)
            scores = self.news_vecss[i][self.cb_indexs] @ user_vecs.T # (n,b) 
            all_scores.append(scores) 
        
        all_scores = np.array(all_scores) # (n_inference,n,b)
        mu = np.mean(all_scores, axis=0) 
        std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
        ucb = mu + std # (n,b) 
        sorted_ids = np.argsort(ucb, axis=0)[-self.rec_batch_size:,:] 
        return self.cb_indexs[sorted_ids], np.empty(0)


class TwoStageNeuralUCB(SingleStageNeuralUCB):
    def __init__(self,device, args, rec_batch_size = 1, n_inference=10, pretrained_mode=True, name='TwoStageNeuralUCB'):
        """Two stage exploration. Use NRMS model. 
            Args:
                rec_batch_size: int, recommendation size. 
                n_inference: int, number of Monte Carlo samples of prediction. 
                pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 

        """
        super(TwoStageNeuralUCB, self).__init__(device, args, rec_batch_size, n_inference, pretrained_mode, name)
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

    def item_rec(self, user_samples, cand_news):
        all_scores = []
        self.model.eval()
        cb_indexs = self._get_cb_news_index(cand_news)
        for i in range(self.n_inference): 
            user_vecs = self._get_user_vecs(self.news_vecss[i], user_samples) # (b,d)
            scores = self.news_vecss[i][cb_indexs] @ user_vecs.T # (n,b) 
            all_scores.append(scores) 
        
        all_scores = np.array(all_scores) # (n_inference,n,b)
        mu = np.mean(all_scores, axis=0) 
        std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
        ucb = mu + std # (n,b) 
        sorted_ids = np.argsort(ucb, axis=0)[-1,:] 
        return cb_indexs[sorted_ids]

    def update(self, context, topics, actions, rewards):
        """Update its internal model. 

        Args:
            context: a user sample
            topics: dummy 
            actions: list of actions; len: rec_batch_size
            rewards: list of rewards; len: rec_batch_size
        """
        # update ts
        print('Updating TS parameters')
        for i, topic in enumerate(topics):  # h_actions are topics
            assert rewards[i] in {0,1}
            self.alphas[topic] += rewards[i]
            self.betas[topic] += 1 - rewards[i] 
        
        # update cb learner
        self.update_learner(context, actions, rewards)

    def sample_actions(self, user_samples): 
        rec_topics = []
        rec_items = []
        self.active_topics = self.cb_topics.copy()
        while len(rec_items) < self.rec_batch_size:
            rec_topic = self.topic_rec()
            rec_topics.append(rec_topic)
            self.active_topics.remove(rec_topic)

            rec_item = self.item_rec(user_samples, self.cb_news[rec_topic])
            rec_items.append(rec_item)
        
        # return [rec_items, rec_topics]
        return np.array(rec_items), np.array(rec_topics)

        
    