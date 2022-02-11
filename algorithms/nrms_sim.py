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
from algorithms.nrms_model import NRMS_Sim_Model
from utils.data_util import read_data, NewsDataset, UserDataset, load_word2vec, SimEvalDataset, SimEvalDataset2, SimTrainDataset, SimValDataset
from utils.metrics import compute_amn

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

        self.nid2index, word2vec, self.nindex2vec = load_word2vec(args)

        # model 
        self.model = NRMS_Sim_Model(word2vec).to(self.device)
        if self.pretrained_mode: # == 'pretrained':
            print('loading a pretrained model from {}'.format(args.sim_path))
            self.model.load_state_dict(torch.load(args.sim_path)) 

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
        rdl = DataLoader(sed, batch_size=batch_size, shuffle=False, num_workers=4) 

        self.model.eval()
        with torch.no_grad():
            scores = []
            for cn in rdl:
                score = self.model(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, compute_loss=False)
                scores.append(torch.sigmoid(score[:,None])) 
            scores = torch.cat(scores, dim=0) 
            print(scores)
            rewards = (scores >= self.threshold).float().detach().cpu().numpy()
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


    def train(self): 
        # TODO: distributed training 
        #   refs: https://horovod.readthedocs.io/en/stable/pytorch.html
        #         https://github.com/Mengyanz/CB4Rec/blob/main/Simulator/run.py 
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = 'runs/nrms_sim_{}'.format(timestamp)
        writer = SummaryWriter(out_path) # https://pytorch.org/docs/stable/tensorboard.html 

        data_path = os.path.join(self.args.root_data_dir, self.args.dataset, 'utils')
        
        with open(os.path.join(data_path, "train_contexts.pkl"), "rb") as fo:
            train_samples = pickle.load(fo)
        train_dataset = SimTrainDataset(self.args, self.nid2index, self.nindex2vec, train_samples) 
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=5)

        with open(os.path.join(data_path, "val_contexts.pkl"), "rb") as fo:
            val_samples = pickle.load(fo)
        val_dataset = SimValDataset(self.args, self.nid2index, self.nindex2vec, val_samples) 
        val_loader = DataLoader(val_dataset, shuffle=False) #, batch_size=self.args.sim_val_batch_size, shuffle=False, num_workers=5)

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
            print(' Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], fscore[ix]))


            writer.add_scalars('Training vs. Val Loss',
                    {'Training': avg_loss, 'Val': avg_vloss}, 
                    epoch_number + 1)

            writer.add_scalars('Per-Imp Metrics',
                    {'auc': imp_metrics_mean[0], 'mrr': imp_metrics_mean[1], 'ndcg5': imp_metrics_mean[2], 'ndcg10': imp_metrics_mean[3], \
                        'ctr': imp_metrics_mean[4]}, 
                    epoch_number + 1)

            writer.add_scalars('Globle Metrics',
                    {'auc': auc, 'mrr': mrr, 'ndcg5': ndcg5, 'ndcg10': ndcg10, 'ctr': ctr, 'best-gmean': gmeans[ix]}, 
                    epoch_number + 1)

            writer.add_scalars('Threshold',
                    {'threshold':thresholds[ix]}, epoch_number + 1)

            writer.flush()

            if imp_metrics_mean[0] > best_vauc: #TODO: select by Global AUC 
                best_vauc = imp_metrics_mean[0] 
                model_path = os.path.join(out_path, 'model_{}'.format(epoch_number))
                torch.save(self.model.state_dict(), model_path)
            
            epoch_number += 1 

def sigmoid(u):
    return 1/(1+np.exp(-u))


def report_metrics(CM):
    tn = CM[0][0]
    tp = CM[1][1]
    fp = CM[0][1]
    fn = CM[1][0]
    acc = np.sum(np.diag(CM)/np.sum(CM))
    sensitivity = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = (2*sensitivity*precision)/(sensitivity+precision)
    return acc, sensitivity, precision, f1
