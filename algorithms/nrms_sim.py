"""Define NRMS simulator. """

import math, os, pickle
import numpy as np 
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
from tqdm import tqdm
import torch 
from torch import nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.simulator import Simulator 
from algorithms.nrms_model import NRMS_Sim_Model, NRMS_IPS_Model
from algorithms.propensity_score import PropensityScore
from utils.data_util import read_data, NewsDataset, UserDataset, load_word2vec, SimEvalDataset, SimEvalDataset2, \
    SimTrainDataset, SimValDataset, SimValWithIPSDataset, SimTrainWithIPSDataset
from utils.metrics import compute_amn

def compute_cdf(x, dist_obj):
    dist = dist_obj[0] 
    params = dist_obj[1]
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    return dist.cdf(x, loc=loc, scale=scale, *arg)

def compute_local_pdf(x, dist_obj, margin):
    return compute_cdf(x + margin, dist_obj) - compute_cdf(x - margin, dist_obj)

class NRMS_Sim(Simulator): 
    def __init__(self, device, args, pretrained_mode=True, name='NRMS_Simulator'): 
        """
        Args:
            pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 
        """
        super(NRMS_Sim, self).__init__(name)

        self.pretrained_mode = pretrained_mode 
        self.name = name 
        self.device = device 
        self.args = args 
        self.threshold = args.sim_threshold

        # preprocessed data 
        # self.nid2index, _, self.news_index, embedding_matrix, self.train_samples, self.valid_samples = read_data(args) 

        self.nid2index, word2vec, self.nindex2vec = load_word2vec(args, utils='utils')
        print('Debug word2vec shape: ', word2vec.shape)

        print('word2vec', word2vec.shape)

        # model 
        self.model = NRMS_Sim_Model(word2vec).to(self.device)
        if self.pretrained_mode: # == 'pretrained':
            print('loading a pretrained model from {}'.format(args.sim_path))
            self.model.load_state_dict(torch.load(os.path.join(args.sim_path, 'model'))) 
            p_dists_fname = os.path.join(args.sim_path, 'p_dists.pkl')
            with open(p_dists_fname, 'rb') as fo: 
                self.p_dists = pickle.load(fo)[0]

            n_dists_fname = os.path.join(args.sim_path, 'n_dists.pkl')
            with open(n_dists_fname, 'rb') as fo: 
                self.n_dists = pickle.load(fo)[0]

            self.sim_margin = args.sim_margin 

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr) 

    def reward(self, uid, news_indexes): 
        """Returns a simulated reward. 
        Args:
            uid: str, user id 
            news_indexes: a list of item index (not nID, but its integer version)
        Return: 
            rewards: (n,m) of 0 or 1 
        """
        batch_size = min(self.args.max_batch_size, len(news_indexes))

        h = self.clicked_history[uid]
        h = h + [0] * (self.args.max_his_len - len(h))
        h = self.nindex2vec[h]

        h = torch.Tensor(h[None,:,:])

        sed = SimEvalDataset2(self.args, news_indexes, self.nindex2vec)
        rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers) 

        self.model.eval()
        with torch.no_grad():
            scores = []
            for cn in rdl:
                score = self.model(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, compute_loss=False)
                scores.append(torch.sigmoid(score[:,None])) 
            scores = torch.cat(scores, dim=0).float().detach().cpu().numpy()
            p_probs = compute_local_pdf(scores, self.p_dists, self.sim_margin) 
            n_probs = compute_local_pdf(scores, self.n_dists, self.sim_margin) 
            
            hard_rewards = (p_probs > n_probs).astype('float') 
            rand_rewards = np.random.binomial(n=1, p = p_probs / (p_probs + n_probs) )
            if self.args.reward_type == 'hard':
                rewards = hard_rewards 
            elif self.args.reward_type == 'soft': 
                rewards = rand_rewards 
            else: 
                rewards = rand_rewards * hard_rewards 
        for s,p,n in zip(scores, p_probs, n_probs):
            print(s,p,n)
        return rewards.ravel()

    def _train_one_epoch(self, epoch_index, train_loader, writer):
        # ref: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
        running_loss = 0 
        last_loss = 0
        for i, batch in enumerate(train_loader): 
            cand_news, clicked_news, targets = batch 
            # Zero gradients for every batch 
            self.optimizer.zero_grad()

            # Make predictions
            loss, score = self.model(cand_news.to(self.device), clicked_news.to(self.device), targets.to(self.device)) 

            # Compute gradients 
            loss.backward()

            # Adjust lr 
            self.optimizer.step()

            # Gather data and report 
            running_loss += loss.item()
            if i % 1000 == 999: 
                last_loss = running_loss / 1000 # loss per batch 
                print(' batch {}/{} loss: {}'.format(i+1, len(train_loader), last_loss))
                tb_x = epoch_index * len(train_loader) + i + 1 
                writer.add_scalar('Loss/train', last_loss, tb_x) 
                running_loss = 0 
        return last_loss 

    def evaluate(self):
        data_path = os.path.join(self.args.root_data_dir, self.args.dataset, 'utils')
        with open(os.path.join(data_path, "val_contexts.pkl"), "rb") as fo:
            val_samples = pickle.load(fo)
        val_dataset = SimValDataset(self.args, self.nid2index, self.nindex2vec, val_samples) 
        val_loader = DataLoader(val_dataset, shuffle=False) 

        self.model.eval() 
        y_scores = [] 
        y_trues = []
        imp_metrics = []
        imp_metrics_prev = []
        with torch.no_grad():
            pbar = tqdm(enumerate(val_loader))
            for i, vdata in pbar: 
                pbar.set_description(' Processing {}/{}'.format(i+1,len(val_loader)))
                cand_news, clicked_news, targets = vdata
                vloss, vscore = self.model(cand_news.to(self.device), clicked_news.to(self.device), targets.to(self.device))
                
                vscore = torch.sigmoid(vscore)

                y_true = targets.cpu().detach().numpy().ravel() 
                y_score = vscore.cpu().detach().numpy().ravel() 


                auc, mrr, ndcg5, ndcg10, ctr = compute_amn(y_true, y_score)
                imp_metrics_prev.append([auc, mrr, ndcg5, ndcg10, ctr])

                p_probs = compute_local_pdf(y_score, self.p_dists, self.sim_margin) 
                n_probs = compute_local_pdf(y_score, self.n_dists, self.sim_margin) 
                y_score = p_probs / (p_probs + n_probs)
                auc, mrr, ndcg5, ndcg10, ctr = compute_amn(y_true, y_score)
                imp_metrics.append([auc, mrr, ndcg5, ndcg10, ctr])

                y_scores.append(y_score) 
                y_trues.append(y_true)


        imp_metrics = np.array(imp_metrics) 
        imp_metrics_prev = np.array(imp_metrics_prev) 

        np.save(os.path.join(self.args.sim_path, 'perimp_metrics'), imp_metrics)
        np.save(os.path.join(self.args.sim_path, 'preds'), y_scores, allow_pickle=True)
        np.save(os.path.join(self.args.sim_path, 'trues'), y_trues, allow_pickle=True)

        print('AUC: transformed {} original {}'.format(np.mean(imp_metrics, axis=0)[0],np.mean(imp_metrics_prev, axis=0)[0]))


    def train(self): 
        # TODO: distributed training 
        #   refs: https://horovod.readthedocs.io/en/stable/pytorch.html
        #         https://github.com/Mengyanz/CB4Rec/blob/main/Simulator/run.py 
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = 'runs/nrms_sim_{}_r{}'.format(timestamp, self.args.sim_npratio)
        writer = SummaryWriter(out_path) # https://pytorch.org/docs/stable/tensorboard.html 

        data_path = os.path.join(self.args.root_data_dir, self.args.dataset, 'utils')
        
        with open(os.path.join(data_path, "train_contexts.pkl"), "rb") as fo:
            train_samples = pickle.load(fo)
        train_dataset = SimTrainDataset(self.args, self.nid2index, self.nindex2vec, train_samples) 
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

        with open(os.path.join(data_path, "val_contexts.pkl"), "rb") as fo:
            val_samples = pickle.load(fo)
        val_dataset = SimValDataset(self.args, self.nid2index, self.nindex2vec, val_samples) 
        val_loader = DataLoader(val_dataset, shuffle=False) #, batch_size=self.args.sim_val_batch_size, shuffle=False, num_workers=self.args.num_workers)

        epoch_number = 0 

        EPOCHS = 10

        best_vauc = 0. 

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            self.model.train() # train mode 
            avg_loss = self._train_one_epoch(epoch_number, train_loader, writer)

            # report mode 
            self.model.eval() 

            running_vloss = 0.0 
            # CM = 0 
            y_scores = [] 
            y_trues = []
            imp_metrics = []
            with torch.no_grad():
                pbar = tqdm(enumerate(val_loader))
                for i, vdata in pbar: 
                    pbar.set_description(' Processing {}/{}'.format(i+1,len(val_loader)))
                    cand_news, clicked_news, targets = vdata
                    vloss, vscore = self.model(cand_news.to(self.device), clicked_news.to(self.device), targets.to(self.device))
                 
                    running_vloss += vloss 
                    vscore = torch.sigmoid(vscore)

                    y_true = targets.cpu().detach().numpy().ravel() 
                    y_score = vscore.cpu().detach().numpy().ravel() 
                    auc, mrr, ndcg5, ndcg10, ctr = compute_amn(y_true, y_score)
                    imp_metrics.append([auc, mrr, ndcg5, ndcg10, ctr])


                    y_scores.append(y_score) 
                    y_trues.append(y_true)

                    # preds = (sigmoid_fn(vscore)>THRESHOLD).float()
                    # CM += confusion_matrix(targets.cpu(), preds.cpu(),labels=[0,1])

            avg_vloss = running_vloss / (i+1) 

            imp_metrics = np.array(imp_metrics) 
            y_scores = np.hstack(y_scores) 
            y_trues = np.hstack(y_trues) 

            fname = os.path.join(out_path, 'scores_labels')
            np.savez(fname, y_scores, y_trues)

            imp_metrics_mean = np.mean(imp_metrics, axis=0)

            auc, mrr, ndcg5, ndcg10, ctr = compute_amn(y_trues.ravel(), y_scores.ravel())

            # Select threshold 
            precision, recall, thresholds = precision_recall_curve(y_trues, y_scores)
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.argmax(fscore)

            print(' LOSS train {:.3f} valid {:.3f}'.format(avg_loss, avg_vloss))
            print(' PER-IMP METRICS auc {:.3f} mrr {:.3f} ndcg5 {:.3f} ndcg10 {:.3f} ctr {:.3f}'\
                .format(imp_metrics_mean[0],imp_metrics_mean[1], imp_metrics_mean[2], imp_metrics_mean[3], imp_metrics_mean[4]))
            print(' GLOBAL METRICS auc {:.3f} mrr {:.3f} ndcg5 {:.3f} ndcg10 {:.3f} ctr {:.3f}'.format(auc, mrr, ndcg5, ndcg10, ctr))
            print(' Best Threshold=%f, fscore=%.3f' % (thresholds[ix], fscore[ix]))


            writer.add_scalars('Training vs. Val Loss',
                    {'Training': avg_loss, 'Val': avg_vloss}, 
                    epoch_number + 1)

            writer.add_scalars('Per-Imp Metrics',
                    {'auc': imp_metrics_mean[0], 'mrr': imp_metrics_mean[1], 'ndcg5': imp_metrics_mean[2], 'ndcg10': imp_metrics_mean[3], \
                        'ctr': imp_metrics_mean[4]}, 
                    epoch_number + 1)

            writer.add_scalars('Globle Metrics',
                    {'auc': auc, 'mrr': mrr, 'ndcg5': ndcg5, 'ndcg10': ndcg10, 'ctr': ctr, 'best-fmscore': fscore[ix]}, 
                    epoch_number + 1)

            writer.add_scalars('Threshold',
                    {'threshold':thresholds[ix]}, epoch_number + 1)

            writer.flush()

            if imp_metrics_mean[0] > best_vauc: #TODO: select by per-imp AUC 
                best_vauc = imp_metrics_mean[0] 
                model_path = os.path.join(out_path, 'model_{}'.format(epoch_number))
                torch.save(self.model.state_dict(), model_path)
            
            epoch_number += 1 


class EmpiricalIPSModel(object):
    def __init__(self, data_path, uid2index):
        self.uid2index = uid2index
        with open(os.path.join(data_path, 'train_pair_count.pkl'), 'rb') as fo: 
            self.train_pair_count = pickle.load(fo)
        self.train_user_count = np.load(os.path.join(data_path, 'train_user_count.npy'))

        with open(os.path.join(data_path, 'val_pair_count.pkl'), 'rb') as fo: 
            self.val_pair_count = pickle.load(fo)
        self.val_user_count = np.load(os.path.join(data_path, 'val_user_count.npy'))

    def compute_ips(self, uids, cand_idxs, train=True): 
        """
        Args:
            uids: (b,) 
            cand_idexs: (n,b)

        Return:
            scores: (b,n)
        """
        b = len(uids)
        n = len(cand_idxs)
        scores = [] 
        # print('uid', len(uids)) 
        # print('cand_idexs', len(cand_idxs)) 
        # for n in cand_idxs:
        #     print(len(n))
        for j in range(b):
            uindex = self.uid2index[uids[j]]
            sub_scores = []
            nids = [cand_idxs[i][j] for i in range(n)]
            for i in nids:
                sub_scores.append(self._compute_ips(uindex,i,train))
            scores.append(sub_scores)
        return torch.Tensor(np.array(scores))

    def _compute_ips(self, uindex, i, train=True):
        if train:
            assert i in self.train_pair_count[uindex]
            nominator = self.train_pair_count[uindex] 
            nominator = 0 if i not in nominator else nominator[i]
            return nominator * 1.0 / self.train_user_count[uindex]
        else:
            assert i in self.val_pair_count[uindex]
            nominator = self.val_pair_count[uindex] 
            nominator = 0 if i not in nominator else nominator[i]
            return nominator * 1.0 / self.val_user_count[uindex]

class NRMS_IPS_Sim(Simulator): 
    def __init__(self, device, args, pretrained_mode=False, name='NRMS_IPS_Sim', train_mode = False): 
        """
        Args:
            pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 
        """
        super(NRMS_IPS_Sim, self).__init__(name)

        self.pretrained_mode = pretrained_mode 
        self.name = name 
        self.device = device 
        self.args = args 
        self.threshold = args.sim_threshold

        # preprocessed data 
        # self.nid2index, _, self.news_index, embedding_matrix, self.train_samples, self.valid_samples = read_data(args) 

        self.nid2index, word2vec, self.nindex2vec = load_word2vec(args)

        print('word2vec', word2vec.shape)

        # model 
        # model 
        self.model = NRMS_IPS_Model(word2vec).to(self.device)
        if self.pretrained_mode: # == 'pretrained':
            print('loading a pretrained model from {}'.format(args.sim_path))
            try:
                self.model.load_state_dict(torch.load(os.path.join(args.sim_path, 'model'))) 
            except:
                self.model.load_state_dict(torch.load(os.path.join(args.sim_path, 'model_fix'))) 
            p_dists_fname = os.path.join(args.sim_path, 'p_dists.pkl')
            with open(p_dists_fname, 'rb') as fo: 
                self.p_dists = pickle.load(fo)[0]

            n_dists_fname = os.path.join(args.sim_path, 'n_dists.pkl')
            with open(n_dists_fname, 'rb') as fo: 
                self.n_dists = pickle.load(fo)[0]

            self.sim_margin = args.sim_margin 

        if train_mode:
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr) 

            print('In train mode, create ips model')
            if self.args.empirical_ips:
                data_path = os.path.join(args.root_data_dir, args.dataset, 'utils')
                with open(os.path.join(data_path, "uid2index.pkl"), "rb") as f:
                    uid2index = pickle.load(f)
                self.ips_model = EmpiricalIPSModel(data_path, uid2index)
            else:
                self.ips_model = PropensityScore(args, device, pretrained_mode=True)
       

    # def reward(self, uid, news_indexes): 
    #     """Returns a simulated reward. 

    #     Args:
    #         uid: str, user id 
    #         news_indexes: a list of item index (not nID, but its integer version)

    #     Return: 
    #         rewards: (n,m) of 0 or 1 
    #     """
    #     batch_size = min(self.args.max_batch_size, len(news_indexes))

    #     h = self.clicked_history[uid]
    #     h = h + [0] * (self.args.max_his_len - len(h))
    #     h = self.nindex2vec[h]

    #     h = torch.Tensor(h[None,:,:])

    #     sed = SimEvalDataset2(self.args, news_indexes, self.nindex2vec)
    #     rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers) 

    #     self.model.eval()
    #     with torch.no_grad():
    #         scores = []
    #         for cn in rdl:
    #             score = self.model(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, None, compute_loss=False)
    #             scores.append(torch.sigmoid(score[:,None])) 
    #         scores = torch.cat(scores, dim=0) 
    #         print(scores)
    #         rewards = (scores >= self.threshold).float().detach().cpu().numpy()
    #     return rewards.ravel()

    def reward(self, uid, news_indexes): 
        """Returns a simulated reward. 

        Args:
            uid: str, user id 
            news_indexes: a list of item index (not nID, but its integer version)

        Return: 
            rewards: (n,m) of 0 or 1 
        """
        batch_size = min(self.args.max_batch_size, len(news_indexes))

        h = self.clicked_history[uid]
        h = h + [0] * (self.args.max_his_len - len(h))
        h = self.nindex2vec[h]

        h = torch.Tensor(h[None,:,:])

        sed = SimEvalDataset2(self.args, news_indexes, self.nindex2vec)
        rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers) 

        self.model.eval()
        with torch.no_grad():
            scores = []
            for cn in rdl:
                score = self.model(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, None,compute_loss=False)
                scores.append(torch.sigmoid(score[:,None])) 
            scores = torch.cat(scores, dim=0).float().detach().cpu().numpy().ravel()
            p_probs = compute_local_pdf(scores, self.p_dists, self.sim_margin) 
            n_probs = compute_local_pdf(scores, self.n_dists, self.sim_margin) 
            
            hard_rewards = (p_probs > n_probs).astype('float') 
            rand_rewards = np.random.binomial(n=1, p = p_probs / (p_probs + n_probs) )
            threshold_rewards = (scores > self.args.sim_threshold).astype('float') 
            if self.args.reward_type == 'hard':
                rewards = hard_rewards 
            elif self.args.reward_type == 'soft': 
                rewards = rand_rewards 
            elif self.args.reward_type == 'hybrid':
                rewards = rand_rewards * hard_rewards 
            elif self.args.reward_type == 'bern':
                rewards = np.random.binomial(n=1, p=scores)
            elif self.args.reward_type == 'threshold_eps':
                rewards = (scores > self.args.sim_threshold).astype('float') 
                EPS = 0.1
                mask = np.random.binomial(n=1, p = np.array([EPS] * rewards.shape[0]))
                # print(mask.shape, rewards.shape)
                assert mask.shape == rewards.shape
                rewards = rewards * (1 - mask) + (1 - rewards) * mask
            elif self.args.reward_type == 'threshold':
                rewards = (scores > self.args.sim_threshold).astype('float')
            else:
                raise NotImplementedError
        # for s,p,n in zip(scores, p_probs, n_probs):
        #    print(s,p,n)
        return rewards.ravel()


    def _train_one_epoch(self, epoch_index, train_loader, writer):
        # ref: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
        running_loss = 0 
        last_loss = 0
        for i, batch in enumerate(train_loader): 
            cand_news, clicked_news, targets, uids, cand_idxs = batch # @TODO: use all news in the impression list
            if not self.args.empirical_ips:
                cand_idxs = [[self.nid2index[n] for n in ns] for ns in cand_idxs]
            ips_scores = self.ips_model.compute_ips(uids, cand_idxs,train=True)  
            # Zero gradients for every batch 
            self.optimizer.zero_grad()

            # Make predictions
            loss, score = self.model(cand_news.to(self.device), clicked_news.to(self.device), targets.to(self.device), ips_scores.to(self.device), normalize=self.args.ips_normalize) 
            # Compute gradients 
            loss.backward()

            # Adjust lr 
            self.optimizer.step()

            # Gather data and report 
            running_loss += loss.item()
            if i % 1000 == 999: 
                last_loss = running_loss / 1000 # loss per batch 
                print(' batch {}/{} loss: {}'.format(i+1, len(train_loader), last_loss))
                tb_x = epoch_index * len(train_loader) + i + 1 
                writer.add_scalar('Loss/train', last_loss, tb_x) 
                running_loss = 0 
        return last_loss 


    def train(self): 
        # TODO: distributed training 
        #   refs: https://horovod.readthedocs.io/en/stable/pytorch.html
        #         https://github.com/Mengyanz/CB4Rec/blob/main/Simulator/run.py 
        if self.args.empirical_ips:
            prefix = 'PROP__empirical_normalize={}'.format(self.args.ips_normalize)
        else:
            prefix = self.args.ips_path.split('/')
            prefix = prefix[-2] + prefix[-1] + '_ipsnormalize={}'.format(self.args.ips_normalize) #datetime.now().strftime('%Y%m%d_%H%M%S')
        
        out_path = 'runs/ipsnrms_{}_R__{}'.format(prefix, self.args.sim_npratio)
        writer = SummaryWriter(out_path) # https://pytorch.org/docs/stable/tensorboard.html 

        data_path = os.path.join(self.args.root_data_dir, self.args.dataset, 'utils')
        
        with open(os.path.join(data_path, "train_contexts.pkl"), "rb") as fo:
            train_samples = pickle.load(fo)
        train_dataset = SimTrainWithIPSDataset(self.args, self.nid2index, self.nindex2vec, train_samples) 
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #, num_workers=self.args.num_workers)

        with open(os.path.join(data_path, "val_contexts.pkl"), "rb") as fo:
            val_samples = pickle.load(fo)
        val_dataset = SimValWithIPSDataset(self.args, self.nid2index, self.nindex2vec, val_samples)  
        val_loader = DataLoader(val_dataset, shuffle=False) #, batch_size=self.args.sim_val_batch_size, shuffle=False, num_workers=self.args.num_workers)

        epoch_number = 0 

        EPOCHS = 10

        best_vauc = 0. 

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            self.model.train() # train mode 
            avg_loss =self._train_one_epoch(epoch_number, train_loader, writer)

            # report mode 
            self.model.eval() 

            running_vloss = 0.0 
            y_scores = [] 
            y_trues = []
            imp_metrics = []
            with torch.no_grad():
                pbar = tqdm(enumerate(val_loader))
                for i, vdata in pbar: 
                    pbar.set_description(' Processing {}/{}'.format(i+1,len(val_loader)))
                    cand_news, clicked_news, targets, uids, cand_idxs = vdata
                    if not self.args.empirical_ips:
                        cand_idxs = [[self.nid2index[n] for n in ns] for ns in cand_idxs]
                    ips_scores = self.ips_model.compute_ips(uids, cand_idxs, train=False)  
                    vloss, vscore = self.model(cand_news.to(self.device), clicked_news.to(self.device), targets.to(self.device), ips_scores.to(self.device), normalize= self.args.ips_normalize)
                 
                    running_vloss += vloss 
                    vscore = torch.sigmoid(vscore)

                    y_true = targets.cpu().detach().numpy().ravel() 
                    y_score = vscore.cpu().detach().numpy().ravel() 
                    auc, mrr, ndcg5, ndcg10, ctr = compute_amn(y_true, y_score)
                    imp_metrics.append([auc, mrr, ndcg5, ndcg10, ctr])


                    y_scores.append(y_score) 
                    y_trues.append(y_true)


            avg_vloss = running_vloss / (i+1) 

            imp_metrics = np.array(imp_metrics) 
            y_scores = np.hstack(y_scores) 
            y_trues = np.hstack(y_trues) 

            fname = os.path.join(out_path, 'scores_labels_ep={}'.format(epoch+1))
            np.savez(fname, y_scores, y_trues)

            imp_metrics_mean = np.mean(imp_metrics, axis=0)

            auc, mrr, ndcg5, ndcg10, ctr = compute_amn(y_trues.ravel(), y_scores.ravel())

            # Select threshold 
            precision, recall, thresholds = precision_recall_curve(y_trues, y_scores)
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.argmax(fscore)

            print(' LOSS train {:.3f} valid {:.3f}'.format(avg_loss, avg_vloss))
            print(' PER-IMP METRICS auc {:.3f} mrr {:.3f} ndcg5 {:.3f} ndcg10 {:.3f} ctr {:.3f}'\
                .format(imp_metrics_mean[0],imp_metrics_mean[1], imp_metrics_mean[2], imp_metrics_mean[3], imp_metrics_mean[4]))
            print(' GLOBAL METRICS auc {:.3f} mrr {:.3f} ndcg5 {:.3f} ndcg10 {:.3f} ctr {:.3f}'.format(auc, mrr, ndcg5, ndcg10, ctr))
            print(' Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], fscore[ix]))


            writer.add_scalars('Training vs. Val Loss',
                    {'Training': avg_loss, 'Val': avg_vloss}, 
                    epoch_number + 1)

            writer.add_scalars('Per-Imp Metrics',
                    {'auc': imp_metrics_mean[0], 'mrr': imp_metrics_mean[1], 'ndcg5': imp_metrics_mean[2], 'ndcg10': imp_metrics_mean[3], \
                        'ctr': imp_metrics_mean[4]}, 
                    epoch_number + 1)

            writer.add_scalars('Globle Metrics',
                    {'auc': auc, 'mrr': mrr, 'ndcg5': ndcg5, 'ndcg10': ndcg10, 'ctr': ctr, 'best-fmscore': fscore[ix]}, 
                    epoch_number + 1)

            writer.add_scalars('Threshold',
                    {'threshold':thresholds[ix]}, epoch_number + 1)

            writer.flush()

            if imp_metrics_mean[0] > best_vauc: #TODO: select by per-imp AUC 
                best_vauc = imp_metrics_mean[0] 
                model_path = os.path.join(out_path, 'model_{}'.format(epoch_number))
                torch.save(self.model.state_dict(), model_path)
            
            epoch_number += 1 

