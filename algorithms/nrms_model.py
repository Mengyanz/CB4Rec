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
                 query_vector_dim = 200,
                 dropout_rate=0.2,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        self.dropout_rate = dropout_rate
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

class NRMS_Model(nn.Module):
    def __init__(self, embedding_matrix):
        super(NRMS_Model, self).__init__()
        self.text_encoder = TextEncoder(embedding_matrix)
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
