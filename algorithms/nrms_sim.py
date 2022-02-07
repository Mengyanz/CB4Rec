"""Define NRMS simulator. """

import math, os, pickle
import numpy as np 
from datetime import datetime
import torch 
import torch.optim as optim 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.simulator import Simulator 
from algorithms.nrms_model import NRMS_Sim_Model
from utils.data_util import read_data, NewsDataset, UserDataset, load_word2vec, SimEvalDataset, SimEvalDataset2, SimTrainDataset, SimValDataset


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

        # preprocessed data 
        # self.nid2index, _, self.news_index, embedding_matrix, self.train_samples, self.valid_samples = read_data(args) 

        self.nid2index, word2vec, self.nindex2vec = load_word2vec(args)

        # model 
        self.model = NRMS_Sim_Model(word2vec).to(self.device)
        if self.pretrained_mode == 'pretrained':
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

        scores = []
        for cn in rdl:
            score = self.model(cn[:,None,:].to(self.device), h.repeat(cn.shape[0],1,1).to(self.device), None, compute_loss=False)
            scores.append(score.detach().cpu().numpy()) 
        scores = np.concatenate(scores)  
        p = sigmoid(scores) 
        rewards = np.random.binomial(size=p.shape, n=1, p=p)
        return rewards 

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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/nrms_sim_{}'.format(timestamp)) # https://pytorch.org/docs/stable/tensorboard.html 

        out_path = os.path.join(self.args.root_data_dir, self.args.dataset, 'utils')
        with open(os.path.join(out_path, "train_contexts.pkl"), "rb") as fo:
            train_samples = pickle.load(fo)
        train_dataset = SimTrainDataset(self.args, self.nid2index, self.nindex2vec, train_samples) 
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)

        with open(os.path.join(out_path, "val_contexts.pkl"), "rb") as fo:
            val_samples = pickle.load(fo)
        val_dataset = SimValDataset(self.args, self.nid2index, self.nindex2vec, val_samples) 
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)


        epoch_number = 0 

        EPOCHS = 5 

        best_vloss = 1_000_000. 

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            self.model.train(True) # train mode 
            avg_loss = self._train_one_epoch(epoch_number, train_loader, writer)

            # report mode 
            self.model.train(False) 

            running_vloss = 0.0 
            for i, vdata in enumerate(val_loader): 
                cand_news, clicked_news, targets = vdata
                vloss, vscore = self.model(cand_news.to(self.device), clicked_news.to(self.device), targets.to(self.device)) 
                running_vloss += vloss 
            avg_vloss = running_vloss / (i+1) 
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            writer.add_scalars('Training vs. Val Loss',
                    {'Training': avg_loss, 'Val': avg_vloss}, 
                    epoch_number + 1)
            writer.flush()

            # Track the best performance and save the model's state 
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss 
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)
            
            epoch_number += 1 

def sigmoid(u):
    return 1/(1+np.exp(-u))