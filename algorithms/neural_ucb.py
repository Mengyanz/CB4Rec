"""Define a simple UCB. """

import math 
import numpy as np 
import torch 
import torch.optim as optim
from torch.utils.data import DataLoader

from core.contextual_bandit import ContextualBanditLearner 
from algorithms.nrms_model import NRMS_Model
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset

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

        # preprocessed data 
        self.nid2index, _, self.news_index, embedding_matrix, self.train_samples, self.valid_samples = read_data(args) 
        # self.news_index: (None, 30) - a set of integers. 

        self.num_news = self.news_index.shape[0] 

        # model 
        self.model = NRMS_Model(embedding_matrix).to(self.device)
        if self.pretrained_mode == 'pretrained':
            self.model.load_state_dict(torch.load(args.learner_path)) 

        print('inferencing news vecs')
        self.news_vecss = []
        for i in range(self.n_inference): 
            news_vecs = self._get_news_vecs() # (n,d)
            self.news_vecss.append(news_vecs)

    def construct_trainable_samples(self, samples, h_actions, h_rewards):
        """construct trainable samples which will be used in NRMS model training

        Args:
            contexts: list of user samples 
            h_actions: (num_context, rec_batch_size,) 
            h_rewards: (num_context, rec_batch_size,) 
        """
        tr_samples = []
        for i, l in enumerate(samples):
            _, _, his, uid, tsp = l
            poss = []
            negs = []
            for j, reward in enumerate(h_rewards[i]):
                if reward == 1:
                    poss.append(h_actions[i][j])
                elif reward == 0:
                    negs.append(h_actions[i][j])
                else:
                    raise Exception("invalid reward")

            if len(poss) > 0:  
                if len(negs) == 0:
                    pass
                    # TODO
                for pos in poss:
                    tr_samples.append([pos, negs, his, uid, tsp])
        return tr_samples

    def update(self, contexts, h_actions, h_rewards):
        """Update its internal model. 

        Args:
            contexts: list of user samples
            h_actions: (num_context, rec_batch_size,) 
            h_rewards: (num_context, rec_batch_size,) 
        """
        print('Updating the internal model of the bandit!')
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        ft_sam = self.construct_trainable_samples(contexts, h_actions, h_rewards)
        ft_ds = TrainDataset(ft_sam, self.nid2index, self.news_index)
        ft_dl = DataLoader(ft_ds, batch_size=self.args.finetune_batch_size, shuffle=True, num_workers=0)
        
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

            # if cnt % 10 == 0:
            #     ft_loader.set_description(f"[{cnt}]steps loss: {loss / (cnt+1):.4f} ")
            #     ft_loader.refresh() 

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
            news_vecs = self.news_vecss[i]
            user_vecs = self._get_user_vecs(news_vecs, user_samples) # (b,d)
            scores = news_vecs @ user_vecs.T # (n,b) 
            all_scores.append(scores) 
        
        all_scores = np.array(all_scores) # (n_inference,n,b)
        mu = np.mean(all_scores, axis=0) 
        std = np.std(all_scores, axis=0) / math.sqrt(self.n_inference) 
        ucb = mu + std # (n,b) 
        sorted_ids = np.argsort(ucb, axis=0)[-self.rec_batch_size:,:] 
        return sorted_ids


class TwoStageNeuralUCB(SingleStageNeuralUCB):
    def __init__(self,device, args, rec_batch_size = 1, n_inference=10, pretrained_mode=True, name='TwoStageNeuralUCB'):
        """Two stage exploration. Use NRMS model. 
            Args:
                rec_batch_size: int, recommendation size. 
                n_inference: int, number of Monte Carlo samples of prediction. 
                pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 

        """
        super(TwoStageNeuralUCB, self).__init__(evice, args, rec_batch_size, n_inference, pretrained_mode, name)
        self.cb_topics = [] # TODO: load from data

    def topic_rec(self):
        """    
        Return
            rec_topic: one recommended topic
        """
        ss =[] 
        for topic in self.active_topics:
            s = np.random.beta(a= self.alpha[topic], b= self.beta[topic])
            ss.append(s)
        rec_topic = self.active_topics[np.argmax(ss)]
        return rec_topic

    def item_rec(self, rec_topic):
        pass

    def sample_actions(self, user_samples): 
        rec_topics = []
        rec_items = []
        self.active_topics = self.cb_topics
        while len(rec_items) < self.rec_batch_size:
            rec_topic = self.topic_rec()
            rec_topics.append(rec_topic)
            self.active_topics.remove(rec_topic)

            rec_item = self.item_rec(rec_topic)
            rec_items.append(rec_item)

        return rec_items

        # for i, topic in enumerate(rec_topics):
        #     assert rewards[i] in {0,1}
        #     self.alphas[topic] += rewards[i]
        #     self.betas[topic] += 1 - rewards[i] 
    