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

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


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
        #       Q, K, V: [bz, seq_len, 300] -> W -> [bz, seq_len, 400]-> q_s: [bz, 20, seq_len, 20]
        batch_size, seq_len, _ = Q.shape

        # print('Q shape: ', Q.shape)
        # print('d_model {}, d_k * n_heads {}'.format(self.d_model, self.d_k * self.n_heads))
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len) 
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) 
        
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask) 
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) 
        return context 

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

class TextEncoder(nn.Module):
    def __init__(self, 
                 embedding_matrix,
                 word_embedding_dim=300, 
                 num_attention_heads=20,
                 attention_dim = 20,
                 query_vector_dim=200,
                 dropout_rate=0.2,
                 news_embedding_dim = 400,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        self.dropout_rate = dropout_rate
        pretrained_news_word_embedding = torch.from_numpy(embedding_matrix).float()
        
        self.word_embedding = nn.Embedding.from_pretrained(
            pretrained_news_word_embedding, freeze=False)
        
        self.multihead_attention = MultiHeadAttention(word_embedding_dim,
                                              num_attention_heads, attention_dim, attention_dim)
        self.additive_attention = AdditiveAttention(             
                                num_attention_heads*attention_dim,
                                query_vector_dim)
        if news_embedding_dim != num_attention_heads * attention_dim:
            self.reduce_dim = True
            self.reduce_dim_linear = nn.Linear(num_attention_heads * attention_dim,
                                           news_embedding_dim)
        else:
            self.reduce_dim = False

    def forward(self, text):
        # REVIEW: remove training=self.training to enable dropout during testing 
        text_vector = F.dropout(self.word_embedding(text.long()),
                                p=self.dropout_rate,
                                training=True
                                )
        multihead_text_vector = self.multihead_attention(
            text_vector, text_vector, text_vector)
        multihead_text_vector = F.dropout(multihead_text_vector,
                                          p=self.dropout_rate,
                                        #   training=self.training
                                        training=True
                                          )
        # batch_size, word_embedding_dim
        text_vector = self.additive_attention(multihead_text_vector)
        if self.reduce_dim:
            text_vector = self.reduce_dim_linear(text_vector)
        return text_vector

class UserEncoder(nn.Module):
    def __init__(self,
                 news_embedding_dim=400,
                 num_attention_heads=20,
                 attention_dim = 20, 
                 query_vector_dim=200
                ):
        super(UserEncoder, self).__init__()
        self.multihead_attention = MultiHeadAttention(news_embedding_dim,
                                              num_attention_heads, attention_dim, attention_dim)
        self.additive_attention = AdditiveAttention(
                                    num_attention_heads*attention_dim,
                                    query_vector_dim)
        
        self.neg_multihead_attention = MultiHeadAttention(news_embedding_dim,
                                                         num_attention_heads, attention_dim, attention_dim)

        if news_embedding_dim != num_attention_heads * attention_dim:
            self.reduce_dim = True
            self.reduce_dim_linear = nn.Linear(num_attention_heads * attention_dim,
                                           news_embedding_dim)
        else:
            self.reduce_dim = False
        
    def forward(self, clicked_news_vecs):
        # print('Debug in user encoder clicked news vecs shape: ', clicked_news_vecs.shape)
        multi_clicked_vectors = self.multihead_attention(
            clicked_news_vecs, clicked_news_vecs, clicked_news_vecs
        )
        pos_user_vector = self.additive_attention(multi_clicked_vectors)
        
        user_vector = pos_user_vector
        if self.reduce_dim:
            user_vector = self.reduce_dim_linear(user_vector)
        return user_vector

class NRMS_Model(nn.Module):
    def __init__(self, embedding_matrix, news_embedding_dim =400):
        super(NRMS_Model, self).__init__()
        self.text_encoder = TextEncoder(embedding_matrix, news_embedding_dim = news_embedding_dim)
        self.user_encoder = UserEncoder(news_embedding_dim=news_embedding_dim)
        
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
        
        
class TopicEncoder(torch.nn.Module):
    def __init__(self, split_large_topic, num_categories=285, reduction_dim=64, dropout_rate=0.2):
        super(TopicEncoder, self).__init__()
        self.num_categories = 312 if split_large_topic else 285
        print("self.num_categories:", self.num_categories)
        self.word_embedding = nn.Embedding(self.num_categories,
                                           reduction_dim)
        self.mlp_head = nn.Sequential(nn.Linear(reduction_dim, reduction_dim),
                                      nn.Dropout(p=dropout_rate),
                                      nn.ReLU(),
                                      nn.Linear(reduction_dim, reduction_dim))
    def forward(self, candidate_news_topicindex):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, reduction_dim
        news_vector = F.dropout(self.word_embedding(candidate_news_topicindex),
                                training=self.training)
        # batch_size, reduction_dim
        final_news_vector = self.mlp_head(news_vector)
        return final_news_vector
    
    def get_topic_embeddings_byindex(self, topic_index):
        topic_indices = torch.LongTensor(topic_index)
        topic_vector = F.dropout(self.word_embedding(topic_indices.to(next(self.word_embedding.parameters()).device)),
                        training=True)
        final_topic_vector = self.mlp_head(topic_vector)
        return final_topic_vector # num_topic, reduction_dim
    def get_all_topic_embeddings(self):
        topic_indices = torch.LongTensor(range(0, self.num_categories)).to(next(self.word_embedding.parameters()).device)
        # topic_indices = torch.LongTensor(topic_order)
        topic_vector = F.dropout(self.word_embedding(topic_indices),
                                training=True)
        final_topic_vector = self.mlp_head(topic_vector)
        return final_topic_vector # num_topic, reduction_dim

class NRMS_Topic_Model(torch.nn.Module):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, embedding_matrix, split_large_topic):
        super(NRMS_Topic_Model, self).__init__()
        self.text_encoder = TextEncoder(embedding_matrix)
        self.user_encoder = UserEncoder()
        self.dimmension_reduction = nn.Sequential(nn.Linear(400, 64),
                                            nn.Dropout(),
                                            nn.ReLU(),
                                            nn.Linear(64, 64))
        self.topic_encoder = TopicEncoder(split_large_topic)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        self.all_topic_vector = None

    def forward(self, candidate_news_topicindex, clicked_news, targets, compute_loss=True):

        # todo: change to topic encoder
        batch_size, npratio = candidate_news_topicindex.shape
        candidate_news_vector = self.topic_encoder(candidate_news_topicindex).view(batch_size, npratio, -1) # batch_size, 1 + K, reduction_dim
        
        batch_size, clicked_news_num, word_num = clicked_news.shape
        clicked_news = clicked_news.view(-1, word_num)
        clicked_news_vector = self.text_encoder(clicked_news).view(batch_size, clicked_news_num, -1) # batch_size, num_clicked_news_a_user, word_embedding_dim
        
        user_vector = self.user_encoder(clicked_news_vector)
        user_vector = self.dimmension_reduction(user_vector) # batch_size, reduction_dim
        # batch_size, 1 + K
        score = torch.bmm(candidate_news_vector, 
                                      user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        if compute_loss:
            loss = self.criterion(torch.sigmoid(score), targets.float())
            return loss, score
        else:
            return score

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, reduction_dim
        """
        # batch_size, reduction_dim
        return self.text_encoder(news)
    
    def get_topic_vector(self, index):
        """
        Args:
            news:
                {
                    "title": batch_size * num
                },
        Returns:
            (shape) batch_size, reduction_dim
        """
        # batch_size, reduction_dim
        return self.topic_encoder(index)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, reduction_dim
        """
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, reduction_dim
            user_vector: reduction_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
        
    def get_all_topic_embedding(self):
        """
        Returns:
            all_topic_vector: topic_num, reduction_dim
        """

        all_topic_vector = self.topic_encoder.get_all_topic_embeddings() # num_topic, reduction_dim

        return all_topic_vector
    
    def get_topic_embeddings_byindex(self, topic_index):
        """
        Returns:
            all_topic_vector: topic_num, reduction_dim
        """

        all_topic_vector = self.topic_encoder.get_topic_embeddings_byindex(topic_index) # num_topic, reduction_dim

        return all_topic_vector

class NRMS_Sim_Model(nn.Module):
    def __init__(self, embedding_matrix):
        super(NRMS_Sim_Model, self).__init__()
        self.text_encoder = TextEncoder(embedding_matrix)
        self.user_encoder = UserEncoder()
        self.m = nn.Sigmoid()
        
        self.criterion = nn.BCELoss() #nn.CrossEntropyLoss()
    
    def forward(self, candidate_news, clicked_news, targets, compute_loss=True):
        """
        Args:
            candidate_news: (batch_size, 1 + npratio, vect_dim)
            clicked_news: (batch_size, max_his_len, vect_dim)
            targets: (batch_size, 1 + npratio)
        """
        batch_size, one_plus_npratio, word_num = candidate_news.shape
        candidate_news = candidate_news.view(-1, word_num)
        candidate_vector = self.text_encoder(candidate_news).view(batch_size, one_plus_npratio, -1)
        
        batch_size, clicked_news_num, word_num = clicked_news.shape
        clicked_news = clicked_news.view(-1, word_num)
        clicked_news_vecs = self.text_encoder(clicked_news).view(batch_size, clicked_news_num, -1)
        
        user_vector = self.user_encoder(clicked_news_vecs)
        
        score = torch.bmm(candidate_vector, user_vector.unsqueeze(-1)).squeeze(dim=-1) # (batch_size,1 + npratio)

        # print(candidate_vector.shape) 
        # print(user_vector.shape)
        # print(user_vector.unsqueeze(-1).shape)
        # print(score.shape)
        # print(batch_size, one_plus_npratio, word_num)
        # print(batch_size, clicked_news_num, word_num)
        
        if compute_loss:
            loss = self.criterion(self.m(score), targets)
            return loss, score
        else:
            return score

class NRMS_IPS_Model(nn.Module):
    def __init__(self, embedding_matrix):
        super(NRMS_IPS_Model, self).__init__()
        self.text_encoder = TextEncoder(embedding_matrix)
        self.user_encoder = UserEncoder()
        
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, candidate_news, clicked_news, targets, ips_scores, compute_loss=True, normalize=True):
        """
        Args:
            candidate_news: (batch_size, 1 + npratio, vect_dim)
            clicked_news: (batch_size, max_his_len, vect_dim)
            targets: (batch_size, 1 + npratio) 
            ips_score: (batch_size, 1 + npratio)
        """
        batch_size, one_plus_npratio, word_num = candidate_news.shape
        candidate_news = candidate_news.view(-1, word_num)
        candidate_vector = self.text_encoder(candidate_news).view(batch_size, one_plus_npratio, -1)
        
        batch_size, clicked_news_num, word_num = clicked_news.shape
        clicked_news = clicked_news.view(-1, word_num)
        clicked_news_vecs = self.text_encoder(clicked_news).view(batch_size, clicked_news_num, -1)
        
        user_vector = self.user_encoder(clicked_news_vecs)
        
        score = torch.bmm(candidate_vector, user_vector.unsqueeze(-1)).squeeze(dim=-1) # (batch_size,1 + npratio)
        
        if compute_loss:
            if normalize:
                norm = torch.sum(1. / ips_scores, axis=1)[:,None]
                loss = (self.loss(score, targets) / ips_scores) / norm # (batch_size, n)
                loss = torch.mean(loss) 
            else:
                loss = torch.mean(self.loss(score, targets) / ips_scores)
            return loss, score
        else:
            return score
