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

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from dataloader import TrainDataset, NewsDataset, UserDataset
from model import NRMS
from metrics import compute_amn, evaluation_split
import pretty_errors

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda:0")
torch.cuda.set_device(device)

def read_data(args, mode = 'train'):
    with open(os.path.join(args.root_data_dir, 'utils', 'nid2index.pkl'), 'rb') as f:
        nid2index = pickle.load(f)
    embedding_matrix = np.load(os.path.join(args.root_data_dir, 'utils', 'embedding.npy'))
    news_index = np.load(os.path.join(args.root_data_dir, 'utils', 'news_index.npy'))

    if mode == 'train':
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/train_sam_uid.pkl'), 'rb') as f:
            train_sam = pickle.load(f)
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/valid_sam_uid.pkl'), 'rb') as f:
            valid_sam = pickle.load(f)

        return nid2index, news_index, embedding_matrix, train_sam, valid_sam
    elif mode == 'test':
        pass

def train(args):

    nid2index, news_index, embedding_matrix, train_sam, valid_sam = read_data(args, 'train')
    train_ds = TrainDataset(args, train_sam, nid2index, news_index)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = NRMS(embedding_matrix).to(device)
    print(model)
    from torchinfo import summary
    output = summary(model, [(args.batch_size, 4, 30), (args.batch_size, 50, 30), (args.batch_size, 4) ], verbose = 0)
    print(str(output).encode('ascii', 'ignore').decode('ascii'))
    raise Exception()
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids=[0,1,2]) 
    # else:
    #     print('single GPU found.')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_auc = 0
    for ep in range(args.epochs):
        loss = 0
        accuary = 0.0
        model.train()
        train_loader = tqdm(train_dl)
        for cnt, batch_sample in enumerate(train_loader):
            candidate_news_index, his_index, label = batch_sample
            sample_num = candidate_news_index.shape[0]
            candidate_news_index = candidate_news_index.to(device)
            his_index = his_index.to(device)
            label = label.to(device)
            bz_loss, y_hat = model(candidate_news_index, his_index, label)

            loss += bz_loss.detach().cpu().numpy()
            optimizer.zero_grad()
            bz_loss.backward()

            optimizer.step()

            if cnt % 10 == 0:
                train_loader.set_description(f"[{cnt}]steps loss: {loss / (cnt+1):.4f} ")
                train_loader.refresh() 
        

        val_scores = eva(args, model, valid_sam, nid2index, news_index)  
        val_auc, val_mrr, val_ndcg, val_ndcg10, ctr = [np.mean(i) for i in list(zip(*val_scores))]
        print(f"[{ep}] epoch auc: {val_auc:.4f}, mrr: {val_mrr:.4f}, ndcg5: {val_ndcg:.4f}, ndcg10: {val_ndcg10:.4f}, ctr: {ctr:.4f}")

        with open(os.path.join(args.model_path, args.dataset, f'{args.dataset}.txt'), 'a') as f:
            f.write(f"[{ep}] epoch auc: {val_auc:.4f}, mrr: {val_mrr:.4f}, ndcg5: {val_ndcg:.4f}, ndcg10: {val_ndcg10:.4f} , ctr: {ctr:.4f}\n")  
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.model_path, args.dataset, f'{args.dataset}.pkl'))
            with open(os.path.join(args.model_path, args.dataset, f'{args.dataset}.txt'), 'a') as f:
                f.write(f"[{ep}] epoch save model\n")
        
def eva(args, model, valid_sam, nid2index, news_index):
    model.eval()
    news_dataset = NewsDataset(news_index)
    news_dl = DataLoader(news_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers)
    news_vecs = []
    for news in tqdm(news_dl):
        news = news.to(device)
        news_vec = model.text_encoder(news).detach().cpu().numpy()
        news_vecs.append(news_vec)
    news_vecs = np.concatenate(news_vecs)

    user_dataset = UserDataset(args, valid_sam, news_vecs, nid2index)
    user_vecs = []
    user_dl = DataLoader(user_dataset, batch_size=1024, shuffle=False, num_workers=args.num_workers)
    for his, tsp in tqdm(user_dl):
        his = his.to(device)
        user_vec = model.user_encoder(his).detach().cpu().numpy()
        user_vecs.append(user_vec)
    user_vecs = np.concatenate(user_vecs)

    val_scores = evaluation_split(news_vecs, user_vecs, valid_sam, nid2index)
    
    return val_scores

if __name__ == "__main__":
    from parameters import parse_args
    args = parse_args()
    train(args)