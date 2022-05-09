"""Define Propensity Score Model. """

import math, os, pickle
import numpy as np 
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch 
from torch import nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.data_util import PropensityScoreDatasetWithRealLabels, load_word2vec, NewsDataset
from utils.metrics import batch_roc_auc_score

class PropensityScoreModel(nn.Module):
    def __init__(self, args, user_dim, item_dim, name='PropensityScoreModel'):
        super(PropensityScoreModel, self).__init__()
        self.name = name 
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.num_news = args.propensity_score_num_pos + args.propensity_score_num_neg 

        self.user_lin = nn.Linear(self.user_dim, 1) 
        self.item_lin = nn.Linear(self.item_dim, 1) 
        self.user_item_lin = nn.Linear(self.user_dim, self.item_dim, bias=False)

        self.loss = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, uvecs, ivecs, labels, mask, compute_loss=True): 
        """
        Args:
            uvecs: (batch_size, d1) 
            ivecs: (batch_size, num_news, d2) 
            labels: (batch_size, num_news) 
            mask: (batch_size, num_news) 

        """
        u = self.user_lin(uvecs) # (batch_size,1)
        i = self.item_lin(ivecs) # (batch_size, num_news, 1)
        ui = self.user_item_lin(uvecs) # (batch_size, d2)
        ui = torch.sum(ui[:,None,:] * ivecs, axis=-1) # (batch_size, num_news)
        score = u + torch.squeeze(i,axis=-1) + ui  # (batch_size, num_news)

        if compute_loss:
            loss = torch.mean(torch.sum(self.loss(score, labels) * mask, axis=1) / torch.sum(mask, axis=1)) 
            # loss = self.loss(score, labels)
            return loss, score 
        else:
            return score

class PropensityScore(object): 
    def __init__(self, args, device, pretrained_mode=False):
        self.args = args 
        self.device = device
        self.pretrained_mode = pretrained_mode
        self.model = PropensityScoreModel(args, user_dim=400, item_dim=400).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr)  

        self.nid2index, self.word2vec, self.nindex2vec = load_word2vec(args)

        self.data_path = os.path.join(args.root_data_dir, args.dataset, 'utils')
        with open(os.path.join(self.data_path, "uid2index.pkl"), "rb") as f:
            self.uid2index = pickle.load(f)

        with open(os.path.join(self.data_path, "train_uid2index.pkl"), "rb") as f:
            self.train_uid2index = pickle.load(f)

        print('Loading user_embs.npy')
        self.user_embs = np.squeeze(np.load(os.path.join(self.data_path, 'user_embs.npy')), axis=1)
        print(self.user_embs.shape)
        print('Loading news_embs.npy')
        self.news_embs = np.load(os.path.join(self.data_path, 'news_embs.npy'))
        print(self.news_embs.shape)

        if self.pretrained_mode: # == 'pretrained':
            print('loading a pretrained IPS model from {}'.format(args.ips_path))
            self.model.load_state_dict(torch.load(args.ips_path)) 

        with open(os.path.join(self.data_path, 'train_pair_count.pkl'), 'rb') as fo: 
            self.train_pair_count = pickle.load(fo)
        self.train_user_count = np.load(os.path.join(self.data_path, 'train_user_count.npy'))

        with open(os.path.join(self.data_path, 'val_pair_count.pkl'), 'rb') as fo: 
            self.val_pair_count = pickle.load(fo)
        self.val_user_count = np.load(os.path.join(self.data_path, 'val_user_count.npy'))

    def compute_ips(self, uid, cand_idx, train=True):
        self.model.eval() 
        with torch.no_grad():
            uvec = torch.Tensor(self.user_embs[[self.uid2index[u] for u in uid]])# (u,d1)
            ivec = np.transpose(np.take(self.news_embs, cand_idx, axis=0), (1,0,2))
            ivec = torch.Tensor(ivec)
            score = self.model(uvec.to(self.device), ivec.to(self.device), None, None, False) #(1,n)
            return torch.sigmoid(score) 

    def _train_one_epoch(self, epoch_index, train_loader, writer):
        running_loss = 0 
        last_loss = 0
        for i, batch_x in tqdm(enumerate(train_loader)): 
            uvec, ivec, labels, mask = batch_x 
            # print(labels)
            self.optimizer.zero_grad()
            loss, score = self.model(uvec.to(self.device), ivec.to(self.device), labels.to(self.device), mask.to(self.device))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99: 
                last_loss = running_loss / 100 # loss per batch 
                print(' batch {}/{} loss: {}'.format(i+1, len(train_loader), last_loss))
                tb_x = epoch_index * len(train_loader) + i + 1 
                writer.add_scalar('Loss/train', last_loss, tb_x) 
                running_loss = 0 
        return last_loss 


    def train_with_resume(self, epoch_number= 0, best_vauc=0, EPOCHS=10, eval=False): 
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = 'runs/propmodel_pn={}-{}'.format(self.args.propensity_score_num_pos, self.args.propensity_score_num_neg)
        writer = SummaryWriter(out_path) 

        if eval:
            EPOCHS = epoch_number + 1

        train_uidset = list(self.train_uid2index)
        uidset = list(self.uid2index)

        if not eval:
            print('Loading user_news_obs.pkl')
            with open(os.path.join(self.data_path, "user_news_obs.pkl"), 'rb') as fo: 
                user_news_obs = pickle.load(fo)
            for u,v in user_news_obs.items():
                user_news_obs[u] = list(set(v))
            train_dataset = PropensityScoreDatasetWithRealLabels(self.args, train_uidset, \
                self.user_embs, self.news_embs, self.nid2index, self.uid2index, user_news_obs, self.train_user_count, self.train_pair_count)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #, num_workers=self.args.num_workers) 

        print('Loading val_user_news_obs.pkl')
        with open(os.path.join(self.data_path, "val_user_news_obs.pkl"), 'rb') as fo: 
            val_user_news_obs = pickle.load(fo)
        for u,v in val_user_news_obs.items():
            val_user_news_obs[u] = list(set(v))

        num_val_users = 10000
        val_dataset = PropensityScoreDatasetWithRealLabels(self.args, uidset[-num_val_users:], self.user_embs, \
            self.news_embs, self.nid2index, self.uid2index, val_user_news_obs, self.val_user_count, self.val_pair_count, rand=False)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False) 
        # print(list(self.nid2index))

        # epoch_number = 0 

        # EPOCHS = 10

        # best_vauc = 0. 

        for epoch in range(epoch_number, EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            self.model.train() # train mode 
            avg_loss = 0 if eval else self._train_one_epoch(epoch_number, train_loader, writer)

            self.model.eval() 

            if not eval:
                model_path = os.path.join(out_path, 'model_{}'.format(epoch_number + 1))
                torch.save(self.model.state_dict(), model_path)

            y_scores = [] 
            y_trues = [] 
            auc_all = [] 
            running_vloss = 0
            # with torch.no_grad():
            #     pbar = tqdm(enumerate(val_loader))
            #     for i, vdata in pbar: 
            #         pbar.set_description(' Processing {}/{}'.format(i+1,len(val_loader)))
            #         v_uvec, v_ivec, v_labels, v_mask = vdata
            #         vloss, vscore = self.model(v_uvec.to(self.device), v_ivec.to(self.device), v_labels.to(self.device), v_mask.to(self.device))
                 
            #         running_vloss += vloss 

            #         y_true = v_labels.cpu().detach().numpy() # (batch_size, num_news)
            #         y_score = vscore.cpu().detach().numpy() # (batch_size, num_news)

            #         print(y_true, y_score)
            #         auc = batch_roc_auc_score(y_true, y_score)
            #         auc_all += auc


            #         y_scores.append(y_score) 
            #         y_trues.append(y_true)


            avg_vloss = 0 # running_vloss / (i+1) 

            # auc_all = np.array(auc_all) 
            # y_scores = np.concatenate(y_scores) 
            # y_trues = np.concatenate(y_trues) 
            # print(y_trues.shape,y_scores.shape)

            # auc_mean = np.mean(auc_all)

            # fname = os.path.join(out_path, 'scores_labels_ep={}'.format(epoch+1))
            # np.savez(fname, y_scores, y_trues)

            print(' LOSS train {:.3f} valid {:.3f}'.format(avg_loss, avg_vloss))
            # print(' AUC {:.3f}'.format(auc_mean))


            writer.add_scalars('Training vs. Val Loss',
                {'Training': avg_loss, 'Val': avg_vloss}, 
                epoch_number + 1)

            writer.flush()
            epoch_number += 1 

            # if auc_mean > best_vauc: #TODO: select by per-imp AUC 
            #     best_vauc = auc_mean 
            #     model_path = os.path.join(out_path, 'model_best_{}'.format(epoch_number))
            #     torch.save(self.model.state_dict(), model_path)