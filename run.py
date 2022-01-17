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
from policy import CB_sim
from metrics import compute_amn, evaluation_split
import pretty_errors

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda:0")
torch.cuda.set_device(device)


def filter_sam(train_sam, valid_sam):
    selected_users = np.load("/home/v-mezhang/blob/data/large/train_valid/selected_users.npy")
    new_train_sam = []
    new_valid_sam = []
    for sam in train_sam:
        uid = sam[3]
        if uid not in selected_users:
            new_train_sam.append(sam)
    for sam in valid_sam:
        uid = sam[3]
        if uid in selected_users:
            new_valid_sam.append(sam)
    return new_train_sam, new_valid_sam


def read_data(args):
    print('loading nid2index')
    with open(os.path.join(args.root_data_dir, args.dataset,  'utils', 'nid2index.pkl'), 'rb') as f:
        nid2index = pickle.load(f)
    print('loading news_info')
    with open(os.path.join(args.root_data_dir, args.dataset,  'utils', 'news_info.pkl'), 'rb') as f:
        news_info = pickle.load(f)
    print('loading embedding')
    embedding_matrix = np.load(os.path.join(args.root_data_dir, args.dataset,  'utils', 'embedding.npy'))
    print('loading news_index')
    news_index = np.load(os.path.join(args.root_data_dir, args.dataset,  'utils', 'news_index.npy'))

    if args.mode == 'train':
        print('loading train_sam')
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/train_sam_uid.pkl'), 'rb') as f:
            train_sam = pickle.load(f)
        print('loading valid_sam')
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/valid_sam_uid.pkl'), 'rb') as f:
            valid_sam = pickle.load(f)

        if args.filter_user:
            print('filtering')
            train_sam, valid_sam = filter_sam(train_sam, valid_sam)

        return nid2index, news_info, news_index, embedding_matrix, train_sam, valid_sam
    elif args.mode == 'test':
        pass
    elif args.mode == 'cb':
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/sorted_train_sam_uid.pkl'), 'rb') as f:
            sorted_train_sam = pickle.load(f)
        with open(os.path.join(args.root_data_dir, args.dataset, 'utils/sorted_valid_sam_uid.pkl'), 'rb') as f:
            sorted_valid_sam = pickle.load(f)

        return nid2index, news_info, news_index, embedding_matrix, sorted_train_sam, sorted_valid_sam

def train(args):

    nid2index, news_info, news_index, embedding_matrix, train_sam, valid_sam = read_data(args)
    train_ds = TrainDataset(args, train_sam, nid2index, news_index)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = NRMS(embedding_matrix).to(device)
    # print(model)
    # from torchinfo import summary
    # output = summary(model, [(args.batch_size, 4, 30), (args.batch_size, 50, 30), (args.batch_size, 4) ], verbose = 0)
    # print(str(output).encode('ascii', 'ignore').decode('ascii'))
    # raise Exception()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0,1,2,3]) 
    else:
        print('single GPU found.')

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
            bz_loss = bz_loss.sum()

            loss += bz_loss.detach().cpu().numpy()
            optimizer.zero_grad()
            bz_loss.backward()

            optimizer.step()

            if cnt % 10 == 0:
                train_loader.set_description(f"[{cnt}]steps loss: {loss / (cnt+1):.4f} ")
                train_loader.refresh() 
        
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

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

def get_cand_news(t, poss, negs, nid2index, m = 10, news_pool = False):
    """
    Generate candidates news based on CTR.

    t: string, impr time
    poss: list, positive samples in impr
    negs: list, neg samples in impr
    m: int, number of candidate news to return

    Return: array, candidate news id 
    """
    poss_nids = np.array([nid2index[n] for n in poss])
    negs_nids = np.array([nid2index[n] for n in negs])

    if news_pool:

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
        return np.concatenate([sample_nids, poss_nids, negs_nids])
    else:
        return np.concatenate([poss_nids, negs_nids])

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


if __name__ == "__main__":
    from parameters import parse_args
    args = parse_args()
    if args.mode == 'train':
        print('mode: train')
        train(args)
    elif args.mode == 'cb':
        cand_nidss = prepare_cand_news(test_sam, test_nid2index)

        # for para in [0, 0.1, 0.2]: # 0.1, '1overt'
        #     print(para)
        #     cb_sim = CB_sim(model_path=model_path, simulator_path=sim_model_path, out_path=model_path, device=device, news_index=test_news_index, nid2index=test_nid2index, sim_flag = sim_flag,
        #     finetune_flag = False, finetune_batch_size = 32, eva_batch_size = 1024)
        #     cb_sim.run_exper(test_sam=test_sam, cand_nidss = cand_nidss, num_exper=10, n_inference = 1, policy='epsilon_greedy', policy_para=para, k = 'all')

        for para in [0.0]: #0.1 - gpu 1; 0.2 - gpu 2; 0.2, 'logt'
            print(para)
            cb_sim = CB_sim(args, device=device)
            cb_sim.run_exper(test_sam=test_sam, cand_nidss=cand_nidss, num_exper=1, n_inference = 1, policy='ucb', policy_para=para, k = 'all')

    # debug
    # nid2index, news_info, news_index, embedding_matrix, train_sam, valid_sam = read_data(args, 'train')
    # print(news_index[nid2index['N10399']])
    # print(news_info['N10399'])
    # pretrained_news_word_embedding = torch.from_numpy(embedding_matrix).float()    
    # word_embedding = nn.Embedding.from_pretrained(
    #     pretrained_news_word_embedding, freeze=False)
    # print(word_embedding(torch.from_numpy(news_index[nid2index['N10399']]).long()))