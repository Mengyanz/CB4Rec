# %%
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
    

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

# %%
device = torch.device("cuda:0")

# %%
torch.cuda.set_device(device)

# %%
dataset = 'small/'

data_path = Path("/home/v-mezhang/blob-plm/data/" + str(dataset) + "utils/")
model_path = Path("/home/v-mezhang/blob-plm/model/" + str(dataset))

date_format_str = '%m/%d/%Y %I:%M:%S %p'

# sys.stdout = open(model_path / 'output.txt', "w")
# print(model_path)
# sys.stdout.flush()

# %%
npratio = 4
max_his_len = 50
min_word_cnt = 3
max_title_len = 30

# %%
batch_size = 32
epoch = 1
lr=0.0001
name = 'nrms_' + dataset[:-1]
retrain = False
online_flag = False
offline_flag = False
cb_flag = True
eva_times = 2

# %% [markdown]
# # collect impressions

# %%
with open(data_path/'train_sam_uid.pkl', 'rb') as f:
    train_sam = pickle.load(f)

with open(data_path/'sorted_train_sam_uid.pkl', 'rb') as f:
    sorted_train_sam = pickle.load(f)
    
with open(data_path/'sorted_valid_sam_uid.pkl', 'rb') as f:
    valid_sam = pickle.load(f)

if os.path.exists(data_path/'test_sam_uid.pkl'):    
    with open(data_path/'test_sam_uid.pkl', 'rb') as f:
        test_sam = pickle.load(f)

# %% [markdown]
# # News Preprocess

# %%
with open(data_path/'nid2index.pkl', 'rb') as f:
    nid2index = pickle.load(f)
    
with open(data_path/'vocab_dict.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

embedding_matrix = np.load(data_path/'embedding.npy')
news_index = np.load(data_path /'news_index.npy')

# %%
if os.path.exists(data_path/'test_nid2index.pkl'):
    with open(data_path/'test_nid2index.pkl', 'rb') as f:
        test_nid2index = pickle.load(f)

    test_news_index = np.load(data_path /'test_news_index.npy')
else: # TODO: for now use valid to do test (cb)
    test_nid2index = nid2index
    test_news_index = news_index
    test_sam = valid_sam

# %%
def cal_ctr(samples, news_click_count, news_impr_count, interval_time,):
    for l in tqdm(samples):
        pos, neg, his, uid, tsp = l
        tsp = datetime.strptime(tsp,date_format_str)
        tidx = int((tsp - start_time).total_seconds()/interval_time) 
        if type(pos) is list:
            for i in pos:
                nidx = nid2index[i]
                news_click_count[nidx, tidx] += 1
                news_impr_count[nidx, tidx] += 1
        else:
            nidx = nid2index[pos]
            news_click_count[nidx, tidx] += 1
            news_impr_count[nidx, tidx] += 1

        for i in neg:
            nidx = nid2index[i]
            news_impr_count[nidx, tidx] += 1
    return news_click_count, news_impr_count

# %% [markdown]
# # News Pool

# %%

interval_time = 3600
start_time =  datetime.strptime(sorted_train_sam[0][-1],date_format_str)
# print(start_time)
end_time = datetime.strptime(valid_sam[-1][-1],date_format_str)
nt = int((end_time - start_time).total_seconds()/interval_time) + 1 
print(len(nid2index))
print(nt)
news_click_count = np.zeros((len(nid2index), nt), dtype=float)
news_impr_count = np.ones((len(nid2index), nt), dtype=float) * 100 # assume 100 times init

news_click_count, news_impr_count = cal_ctr(train_sam, news_click_count, news_impr_count, interval_time)
news_click_count, news_impr_count = cal_ctr(valid_sam, news_click_count, news_impr_count, interval_time)

# %%
# news_ctr = np.zeros_like(news_click_count)
# for i in tqdm(range(news_click_count.shape[0])):
#     for j in range(news_click_count.shape[1]):
#         if news_impr_count[i,j] == 0:
#             assert news_click_count[i,j] == 0
#             news_ctr[i,j] = 0
#         else:
#             news_ctr[i,j] = news_click_count[i,j]/news_impr_count[i,j]
news_ctr = news_click_count/news_impr_count

# %%
# import matplotlib.pyplot as plt
# news_ctr = news_click_count/news_impr_count
# plt.imshow(news_ctr[:,166])
# plt.colorbar()
tidx = 111
nonzero = news_ctr[:,tidx][news_ctr[:, tidx] > 0]
len(nonzero)

# %%
nonzero_count = []
for i in range(news_click_count.shape[1]):
    nonzero = news_ctr[:,i][news_ctr[:, i] > 0]
    nonzero_count.append(len(nonzero))
# plt.hist(nonzero_count)

# %% [markdown]
# # Dataset & DataLoader

# %%
def newsample(nnn, ratio):
    if ratio > len(nnn):
        return nnn + ["<unk>"] * (ratio - len(nnn))
    else:
        return random.sample(nnn, ratio)

# %%
class TrainDataset(Dataset):
    def __init__(self, samples, nid2index, news_index):
        self.news_index = news_index
        self.nid2index = nid2index
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # pos, neg, his, neg_his
        pos, neg, his, uid, tsp = self.samples[idx]
        neg = newsample(neg, npratio)
        
        candidate_news = [pos] + neg
        # print('pos: ', pos)
        # for n in candidate_news:
        #     print(n)
        #     print(self.nid2index[n])
        if type(candidate_news[0]) is str:
            assert candidate_news[0].startswith('N') # nid
            candidate_news = self.news_index[[self.nid2index[n] for n in candidate_news]]
        else: # nindex
            # print(candidate_news)
            # TODO: check why float n appears in candidate news
            candidate_news = self.news_index[[int(n) for n in candidate_news]]
        his = [self.nid2index[n] for n in his] + [0] * (max_his_len - len(his))
        his = self.news_index[his]
        
        label = np.array(0)
        return candidate_news, his, label

# %%
class NewsDataset(Dataset):
    def __init__(self, news_index):
        self.news_index = news_index
        
    def __len__(self):
        return len(self.news_index)
    
    def __getitem__(self, idx):
        return self.news_index[idx]

# %%
news_dataset = NewsDataset(news_index)

# %%
class UserDataset(Dataset):
    def __init__(self, 
                 samples,
                 news_vecs,
                 nid2index):
        self.samples = samples
        self.news_vecs = news_vecs
        self.nid2index = nid2index
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        poss, negs, his, uid, tsp = self.samples[idx]
        his = [self.nid2index[n] for n in his] + [0] * (max_his_len - len(his))
        his = self.news_vecs[his]
        return his, tsp

# %% [markdown]
# # NRMS

# %%
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True)  + 1e-8)
        
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model # 300
        self.n_heads = n_heads # 20
        self.d_k = d_k # 20
        self.d_v = d_v # 20
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads) # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads) # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads) # 300, 400
        
        self._initialize_weights()
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                
    def forward(self, Q, K, V, attn_mask=None):
        residual, batch_size = Q, Q.size(0)
        
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, max_len, max_len) 
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) 
        
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask) 
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) 
        return context 

# %%

class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg: 
        d_h: the last dimension of input
    '''
    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x

# %%
class TextEncoder(nn.Module):
    def __init__(self, 
                 word_embedding_dim=300, 
                 num_attention_heads=20,
                 query_vector_dim = 200,
                 dropout_rate=0.2,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        self.dropout_rate = 0.2
        pretrained_news_word_embedding = torch.from_numpy(embedding_matrix).float()
        
        self.word_embedding = nn.Embedding.from_pretrained(
            pretrained_news_word_embedding, freeze=False)
        
        self.multihead_attention = MultiHeadAttention(word_embedding_dim,
                                              num_attention_heads, 20, 20)
        self.additive_attention = AdditiveAttention(num_attention_heads*20,
                                                    query_vector_dim)
    def forward(self, text):
        # REVIEW: remove training=self.training to enable dropout during testing 
        text_vector = F.dropout(self.word_embedding(text.long()),
                                p=self.dropout_rate,
                                # training=self.training
                                )
        multihead_text_vector = self.multihead_attention(
            text_vector, text_vector, text_vector)
        multihead_text_vector = F.dropout(multihead_text_vector,
                                          p=self.dropout_rate,
                                        #   training=self.training
                                          )
        # batch_size, word_embedding_dim
        text_vector = self.additive_attention(multihead_text_vector)
        return text_vector

# %%
class UserEncoder(nn.Module):
    def __init__(self,
                 news_embedding_dim=400,
                 num_attention_heads=20,
                 query_vector_dim=200
                ):
        super(UserEncoder, self).__init__()
        self.multihead_attention = MultiHeadAttention(news_embedding_dim,
                                              num_attention_heads, 20, 20)
        self.additive_attention = AdditiveAttention(num_attention_heads*20,
                                                    query_vector_dim)
        
        self.neg_multihead_attention = MultiHeadAttention(news_embedding_dim,
                                                         num_attention_heads, 20, 20)
        
    def forward(self, clicked_news_vecs):
        multi_clicked_vectors = self.multihead_attention(
            clicked_news_vecs, clicked_news_vecs, clicked_news_vecs
        )
        pos_user_vector = self.additive_attention(multi_clicked_vectors)
        
        user_vector = pos_user_vector
        return user_vector

# %%
class NRMS(nn.Module):
    def __init__(self):
        super(NRMS, self).__init__()
        self.text_encoder = TextEncoder()
        self.user_encoder = UserEncoder()
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, candidate_news, clicked_news, targets, compute_loss=True):
        batch_size, npratio, word_num = candidate_news.shape
        candidate_news = candidate_news.view(-1, word_num)
        candidate_vector = self.text_encoder(candidate_news).view(batch_size, npratio, -1)
        
        batch_size, clicked_news_num, word_num = clicked_news.shape
        clicked_news = clicked_news.view(-1, word_num)
        clicked_news_vecs = self.text_encoder(clicked_news).view(batch_size, clicked_news_num, -1)
        
        user_vector = self.user_encoder(clicked_news_vecs)
        
        score = torch.bmm(candidate_vector, user_vector.unsqueeze(-1)).squeeze(dim=-1)
        
        if compute_loss:
            loss = self.criterion(score, targets)
            return loss, score
        else:
            return score

# %% [markdown]
# # Train

# %%
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

# %%
def compute_amn(y_true, y_score):
    auc = roc_auc_score(y_true,y_score)
    mrr = mrr_score(y_true,y_score)
    ndcg5 = ndcg_score(y_true,y_score,5)
    ndcg10 = ndcg_score(y_true,y_score,10)
    return auc, mrr, ndcg5, ndcg10

def evaluation_split(news_vecs, user_vecs, samples, nid2index):
    all_rslt = []
    for i in tqdm(range(len(samples))):
        poss, negs, _, _, _ = samples[i]
        user_vec = user_vecs[i]
        y_true = [1] * len(poss) + [0] * len(negs)
        news_ids = [nid2index[i] for i in poss + negs]
        news_vec = news_vecs[news_ids]
        y_score = np.multiply(news_vec, user_vec)
        y_score = np.sum(y_score, axis=1)
        try:
            all_rslt.append(compute_amn(y_true, y_score))
        except Exception as e:
            print(e)
    return np.array(all_rslt)


# %% [markdown]
# # Bandit Simulation

# %%
class CB_sim():
    def __init__(
        self, model_path, simulator_path, out_path, device,
        news_index, nid2index,
        finetune_batch_size = 32, eva_batch_size = 1024, dropout_flag = True
    ):
        self.model = NRMS().to(device)
        self.model.load_state_dict(torch.load(model_path/f'{name}.pkl'))
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(self.model, device_ids=[0,1,2]) 
        else:
            print('single GPU found.')
        
        # TODO: change simulator to PLM
        self.simulator = NRMS().to(device)
        self.simulator.load_state_dict(torch.load(simulator_path/f'{name}.pkl'))

        self.news_index = news_index
        self.nid2index = nid2index

        self.out_path = out_path
        self.dropout_flag = dropout_flag
        
        self.finetune_batch_size = finetune_batch_size
        self.eva_batch_size = eva_batch_size
        self.date_format_str = '%m/%d/%Y %I:%M:%S %p'

    
    def enable_dropout(self):
        for m in self.model.modules():
            if m.__class__.__name__.startswith('dropout'):
                print(m)
                m.train() 
        
    def get_news_vec(self, model, batch_news_index):
        batch_news_dataset = NewsDataset(batch_news_index)
        news_dl = DataLoader(batch_news_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
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
        user_dl = DataLoader(user_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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
        ft_dl = DataLoader(ft_ds, batch_size=self.finetune_batch_size, shuffle=True, num_workers=0)
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
        if self.policy_para == 'logt':
            # REVIEW: whether to use batch id?
            para = np.log(batch_id + 2) * 0.1
            print('policy para: ', para)
        else:
            para = self.policy_para
        ucb_score = mean + para * std

        rec_nids = np.argsort(ucb_score)[-self.k:]
        return rec_nids

    def epsilon_greedy(self, exper_id, batch_id, y_score):
        rec = []
        y_score = y_score[0]
        p = np.random.rand(self.k)
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
            greedy_nids = np.argsort(y_score)[-n_greedy:]

        if n_greedy < self.k:
            if len(list(set(list(range(len(y_score)))) - set(greedy_nids))) == 0:
                print('y_score: ', y_score)
                print('n_greedy: ', n_greedy)
                print(np.array(list(set(list(range(len(y_score)))) - set(greedy_nids))))

            eps_nids = np.random.choice(np.array(list(set(list(range(len(y_score)))) - set(greedy_nids))), size = self.k - n_greedy, replace=False)
            rec_nids= np.concatenate([greedy_nids, eps_nids])
            return rec_nids
        else:
            return greedy_nids 
        
    def sigmoid(self, x):
        return np.array([1/(1 + math.exp(-i)) for i in x])

    def get_opt_set(self, sam, cand):
        """
        prepare candidate news and get optimal set 
        """
        opts = []
        sim_news_vecs = self.get_news_vec(self.simulator, self.news_index)
        sim_user_vecs = self.get_user_vec(self.simulator, sam, sim_news_vecs, self.nid2index) 

        for i in range(len(sam)):
            poss, negs, _, uid, tsq = sam[i] 
            
            sim_user_vec = sim_user_vecs[i]
            sim_news_vec = sim_news_vecs[cand[i]]
            sim_y_score = np.sum(np.multiply(sim_news_vec, sim_user_vec), axis=1)
            # REVIEW: how to choose opt set
            # assume the user would at most click half of the recommended news
            # opt_nids = np.argsort(sim_y_score)[-max(int(self.k/2), 1):]
            opt_nids = np.argsort(sim_y_score)[-max(int(len(sim_y_score)/2), 1):]  
            opt_nids = opt_nids[self.sigmoid(sim_y_score[opt_nids]) > 0.5]
            # print(opt_nids)

            opts.append(opt_nids) 
        return opts

    def get_reward(self, batch_sam, batch_recs):
        rewards = []
        sim_news_vecs = self.get_news_vec(self.simulator, self.news_index)
        sim_user_vecs = self.get_user_vec(self.simulator, batch_sam, sim_news_vecs, self.nid2index) 

        for i in range(len(batch_sam)):
            poss, negs, _, _, tsq = batch_sam[i] 
            
            sim_user_vec = self.sim_user_vecs[i]
            sim_news_vec = self.sim_news_vecs[batch_recs[i]]
            sim_y_score = np.sum(np.multiply(sim_news_vec, sim_user_vec), axis=1)
            # assume the user would at most click half of the recommended news
            reward = 1 if self.sigmoid(sim_y_score) > 0.5 else 0
            # opt_nids = opt_nids[self.sigmoid(sim_y_score[opt_nids]) > 0.5]
            # print(opt_nids)

            rewards.append(reward)
        return rewards
        
    def rec(self, batch_sam, batch_cand, opts, batch_id, exper_id):
        """
        Simulate recommendations
        """
        if self.policy == 'epsilon_greedy':
            self.n_inference = 1
        
        self.model.eval()
        if self.dropout_flag:
            self.enable_dropout()

        y_scores = defaultdict(list) # key: sam_id, value: list of n_inference scores 
        start_id = self.eva_batch_size * batch_id

        with torch.no_grad():
            
            """
            inf_idx, behav_idx, user_idx, news_idx
            1, 1, 1, 1
            1, 1, 1, 2
            1, 1, 1, 3
            1, 2, 1, 4

            n_inf, n1, user_dim
            n_inf, n1, news_dim

            n_inf, n1 = 
            """
            # generate scores with uncertainty (dropout during inference)
            for _ in tqdm(range(self.n_inference)):
                # TODO: speed up - only pass batch news index
                # inf_time1 = time.time()
                news_vecs = self.get_news_vec(self.model, self.news_index) # batch_size * news_dim (400)
                user_vecs = self.get_user_vec(self.model, batch_sam, news_vecs, self.nid2index)
                # inf_time11 = time.time()
                # print('single time news encoder inference: ', inf_time11-inf_time1)
                
                for i in range(len(batch_sam)):
                    t = start_id + i
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
                rec_nids = self.get_ucb_score(exper_id, batch_id, y_scores[t])
            elif self.policy == 'epsilon_greedy':
                rec_nids = self.epsilon_greedy(exper_id, batch_id, y_scores[t])
            else:
                raise NotImplementedError

            self.recs[exper_id].append(rec_nids) 

            opt_nids = opts[t]

            assert len(set(rec_nids)) == self.k
            # assert len(set(opt_nids)) == self.k

            reward = len(set(rec_nids) & set(opt_nids)) # reward as the overlap between rec and opt set
            # self.cumu_reward += reward
            self.rewards[exper_id, t] = reward
            # TODO: next line only for single arm rec
            self.opt_rewards[exper_id, t] = 1 if len(set(opt_nids)) > 0 else 0

            # REVIEW: put opt nids in rec poss as well?
            rec_poss = [rec_nid for rec_nid in rec_nids if rec_nid in opt_nids]
            # rec_poss = [rec_nid for rec_nid in set(rec_nids).union(opt_nids)]
            # print('rec poss: ', rec_poss) 
            rec_negs = list(set(batch_cand[i]) - set(rec_poss))
            
            self.rec_sam.append([rec_poss, rec_negs, his, uid, tsq])
        
    def run_exper(self, test_sam, cand_nidss, num_exper = 10, n_inference = 2, policy = 'ucb', policy_para = 0.1, k = 1):
        
        num_sam = len(test_sam)
        n_batch = math.ceil(float(num_sam)/self.eva_batch_size)
        self.rec_sam = []
        self.rewards = np.zeros((num_exper, num_sam))
        self.opt_rewards = np.zeros((num_exper, num_sam))
        self.recs = defaultdict(list)

        self.n_inference = n_inference
        self.policy = policy
        self.policy_para = policy_para
        self.k = k # num_rec

        update_time = None
        update_batch = 0

        opts = self.get_opt_set(test_sam, cand_nidss)
        

        for j in range(num_exper):
            # self.cumu_reward = 0
            print('exper: ', j)
            
            for i in range(n_batch):
                upper_range = min(self.eva_batch_size * (i+1), len(test_sam))
                batch_sam = test_sam[self.eva_batch_size * i: upper_range]
                batch_cand = cand_nidss[self.eva_batch_size * i: upper_range]

                self.rec(batch_sam, batch_cand, opts, i, j)

                if update_time is None:
                    update_time = datetime.strptime(str(batch_sam[0][-1]), self.date_format_str)
                    print('init time: ', update_time)

                batch_time = datetime.strptime(str(batch_sam[-1][-1]), self.date_format_str)
                if (batch_time- update_time).total_seconds() > 3600:
                    lower_range = self.eva_batch_size * update_batch
                    ft_sam = self.rec_sam[lower_range: upper_range]
                    if upper_range - lower_range > 512:
                        print('finetune with: '  + str(lower_range) + ' ~ ' + str(upper_range))
                        self.finetune(ft_sam=ft_sam)

                        update_time = batch_time
                        update_batch = i + 1
                        print('update at: ', update_time)
                    else: 
                        print('no finetune due to insufficient samples: ', str(upper_range - lower_range))
            update_time = None
            update_batch = 0

        self.save_results()

    def save_results(self):
        folder_name = 'rec' + str(self.k)
        save_path = os.path.join(self.out_path, folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        policy_name = self.policy + '_' + str(self.policy_para)
        torch.save(self.model.state_dict(), os.path.join(save_path, (policy_name + f'_{name}_finetune.pkl')))
        # with open(os.path.join(self.out_path, (policy_name + "_recs.pkl")), "wb") as f:
        #     pickle.dump(self.recs, f)
        # with open(os.path.join(self.out_path, (policy_name + "_opts.pkl")), "wb") as f:
        #     pickle.dump(self.opts, f)
        with open(os.path.join(save_path, (policy_name + "_rewards.pkl")), "wb") as f:
            pickle.dump(self.rewards, f)
        with open(os.path.join(save_path, (policy_name+ "_opt_rewards.pkl")), "wb") as f:
            pickle.dump(self.opt_rewards, f)


def get_cand_news(t, poss, negs, nid2index, m = 10):
    """
    Generate candidates news based on CTR.

    t: string, impr time
    poss: list, positive samples in impr
    negs: list, neg samples in impr
    m: int, number of candidate news to return

    Return: array, candidate news id 
    """
    t = datetime.strptime(t,date_format_str)
    tidx = int((t - start_time).total_seconds()/interval_time)

    nonzero = news_ctr[:,tidx -1][news_ctr[:, tidx-1] > 0]
    nonzero_idx = np.where(news_ctr[:, tidx-1] > 0)[0]
    # print(nonzero_idx)

    nonzero = nonzero/nonzero.sum()
    assert (nonzero.sum() - 1) < 1e-3
    # print(np.sort(nonzero)[-5:])

    # sampling according to normalised ctr in last one hour
    sample_nids = np.random.choice(nonzero_idx, size = m, replace = False, p = nonzero)
    # REVIEW: check whether the sampled nids are reasonable
    
    poss_nids = np.array([nid2index[n] for n in poss])
    negs_nids = np.array([nid2index[n] for n in negs])

    return np.concatenate([sample_nids, poss_nids, negs_nids])

def prepare_cand_news(sam, nid2index):
    """
    prepare candidate news and get optimal set 
    """
    cand_nidss = []

    for i in range(len(sam)):
        poss, negs, _, _, tsq = sam[i] 
        cand_nids = get_cand_news(tsq, poss, negs, nid2index)
        cand_nidss.append(np.array(cand_nids))      

    return cand_nidss


# %%

cand_nidss = prepare_cand_news(test_sam, test_nid2index)

# for para in [0, 0.1, '1overt']: # gpu 0
#     print(para)
#     cb_sim = CB_sim(model_path=model_path, simulator_path=model_path, out_path=model_path, device=device, news_index=test_news_index, nid2index=test_nid2index,
#     finetune_batch_size = 32, eva_batch_size = 1024)
#     cb_sim.run_exper(test_sam=test_sam, cand_nidss = cand_nidss, num_exper=10, n_inference = 1, policy='epsilon_greedy', policy_para=para, k = 1)

for para in [0.1, 'logt']: #0.1 - gpu 1; 0.2 - gpu 2;
    print(para)
    cb_sim = CB_sim(model_path=model_path, simulator_path=model_path, out_path=model_path, device=device, news_index=test_news_index, nid2index=test_nid2index,
    finetune_batch_size = 32, eva_batch_size = 1024)
    cb_sim.run_exper(test_sam=test_sam, cand_nidss=cand_nidss, num_exper=10, n_inference = 10, policy='ucb', policy_para=para, k = 1)



