"""Define a linear ucb recommendation policy. """

import math 
import numpy as np 
from collections import defaultdict
import torch 
import os
from torch import nn
from core.contextual_bandit import ContextualBanditLearner 
from algorithms.linucb import LinUCB
from algorithms.naive_linucb import NaiveLinUCB
from utils.data_util import read_data, NewsDataset, UserDataset, TrainDataset, load_word2vec, load_cb_topic_news, SimEvalDataset, SimEvalDataset2, SimTrainDataset
        

        

class LinUCBBaseStruct():
    def __init__(self,dim,init,alpha):
        self.dim = dim
        self.A = np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))  
        if init!='zero':
            self.theta = np.random.rand(self.dim)
        else:
            self.theta = np.zeros(self.dim)
        self.alpha = alpha # store the new alpha calcuated in each iteratioin

    def getProb(self,fv):
        if self.alpha==-1:
            raise AssertionError
        mean = fv.dot(self.theta)
        var = np.array([np.sqrt(x.dot(self.Ainv).dot(x.T)) for x in fv])
        pta=mean+self.alpha*var
        return pta

    def getInv(self, old_Minv, nfv):
        # new_M=old_M+nfv*nfv'
        # try to get the inverse of new_M
        tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
        tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
        new_Minv=old_Minv-tmp_a/tmp_b
        return new_Minv

    def updateParameters(self, a_fv, reward):
        self.A+=a_fv.dot(a_fv.T)
        self.b+=a_fv * reward
        self.Ainv=self.getInv(self.Ainv,a_fv)
        self.theta=np.dot(self.Ainv, self.b)


class LinUCBUserStruct():
    def __init__(self,uid,root, args):
        self.uid = uid
        self.root = root
        self.base_linucb = {}
        self.path = []
        # if not cfg.random_choice:
        self.linucb = NaiveLinUCB(args)

class Item():
    def __init__(self,gid,x1):
        self.gid     = gid
        self.fv      = {} #feature vector for training/simulator
        self.fv['t'] = x1 #training

class HCB(ContextualBanditLearner):
    def __init__(self,device, args,root, name='HCB'):
        
        super(HCB, self).__init__(args, device, name)
        self.device = device
        self.args = args
        self.name = name
        self.items = {} 
        self.n_items = 0
        self.users = {}
        try:
            self.alpha = self.args.gamma
        except:
            self.alpha = -1
        self.root = root
        self.total_depth = 2
        
        self.load_items()

    def load_items(self):
        nindex2embedding = np.load(os.path.join(self.args.root_data_dir, self.args.dataset,  'utils', 'nindex2embedding.npy'))
        for gid in range(len(nindex2embedding)):
            x1 = nindex2embedding[gid]
            self.items[gid] = Item(gid,x1)
        self.n_items=len(self.items)
        
    def sample_actions(self,uid,root=None):
        
        score_budget = self.per_rec_score_budget * self.rec_batch_size
        try:
            user=self.users[uid]
        except:
            self.dim = self.root.emb.shape[0]
            self.users[uid]=LinUCBUserStruct(uid,self.root,self.args)
            user=self.users[uid]
        current_node = user.root
        user.path = []
        
        if len(user.base_linucb) == 0 and self.pretrained_mode and len(self.clicked_history[uid]) > 0:
            for depth in range(self.total_depth):
                children = current_node.children
                baselinucb = LinUCBBaseStruct(self.dim,"zero",self.alpha)
                if depth not in user.base_linucb:
                    if self.args.one_linucb_perlayer:
                        user.base_linucb[depth] = baselinucb
                    else:
                        user.base_linucb[depth] = LinUCBBaseStruct(self.dim,"zero",self.alpha)
            for gid in self.clicked_history[uid]:
                for _, arm in user.root.children.items():
                    if gid in arm.gids:
                        user.path.append(arm)
                        for _, arm_2 in arm.children.items():
                            if gid in arm_2.gids:
                                user.path.append(arm_2)
                                break
                        break
                if len(user.path) == 0:
                    continue
                
                picked_arms = self.items[gid]
                self.update(None, [picked_arms], [1], mode='item', uid=uid)
                user.path = []
                
        depth = 0
        while(current_node.is_leaf==False):
            children = current_node.children
            max_r = float('-inf')
            aid = None
            baselinucb = LinUCBBaseStruct(self.dim,"zero",self.alpha)
            if depth not in user.base_linucb:
                if self.args.one_linucb_perlayer:
                    user.base_linucb[depth] = baselinucb
                else:
                    user.base_linucb[depth] = LinUCBBaseStruct(self.dim,"zero",self.alpha)
            poolsize = int(score_budget/self.total_depth)
            if len(children)<=poolsize:
                arms = [children[i] for i in range(len(children))]
            else:
                aids = np.random.choice(len(children),poolsize,replace=False)
                arms = [children[i] for i in aids]
                
            arms_emb = np.array([arm.emb for arm in arms])
            ucb = user.base_linucb[depth].getProb(arms_emb)
            aid_argmax = np.argsort(ucb)[::-1][:1].tolist()[0]
            # for index,arm in enumerate(arms):
            #     reward = user.base_linucb[depth].getProb(arm.emb, self.rec_batch_size)
            #     if reward>max_r:
            #         aid = index
            #         max_r = reward
            arm_picker = arms[aid_argmax]
            user.path.append(arm_picker)
            current_node = arm_picker
            depth += 1
            
        if len(arm_picker.gids)<=self.args.per_rec_score_budget:
            arms = [self.items[gid] for gid in arm_picker.gids]
        else:
            arms = [self.items[gid] for gid in np.random.choice(arm_picker.gids,score_budget,replace=False).tolist()]
        
        picked_arms = self.users[user.uid].linucb.decide(user.uid,arms, self.rec_batch_size)
        # items = [arm.gid for arm in arms]
        return np.empty(0), picked_arms

    def update(self,topics,picked_arms,feedbacks, mode,uid = None):
        if mode == 'item':
            # print('Update hcb parameters for user {}!'.format(uid))
            user = self.users[uid]
            path = user.path
            assert len(path)!=0
            for i,arm_picker in enumerate(path):
                depth = i
                user.base_linucb[depth].updateParameters(arm_picker.emb, int(np.sum(feedbacks))) # or np.any
            for arm, feedback in zip(picked_arms, feedbacks):
                user.linucb.updateParameters(arm,feedback,user.uid)
            
            
    def update_clicked_history(self, pos, uid):
        """
        Args:
            pos: a list of str nIDs, positive news of uid 
            uid: str, user id 
        """
        # DO NOT UPDATE CLICKED HISTORY
        pass 

    def update_data_buffer(self, pos, neg, uid, t): 
        # for nid in pos:
        #     self.D[uid].append(nid)
        #     self.c[uid].append(1)
        # for nid in neg:
        #     self.D[uid].append(nid)
        #     self.c[uid].append(0)
        # print('size(data_buffer): {}'.format(len(self.D)))
        pass
    
    def reset(self):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        # self.D = defaultdict(list) 
        # self.c = defaultdict(list)
        self.users = {}