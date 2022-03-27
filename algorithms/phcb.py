import numpy as np
import math
import os
from collections import defaultdict
from core.contextual_bandit import ContextualBanditLearner 
from algorithms.naive_linucb import NaiveLinUCB

class ArmStruct():
    def __init__(self,arm,depth, args):
        self.args = args
        self.arm = arm
        self.gids = arm.gids
        self.depth = depth # only when arm is a tree node
        self.itemclick = defaultdict(bool)
        self.feedback = defaultdict(float)
        self.vv = defaultdict(int)

    def expand(self):
        if (sum(self.vv.values()))<self.args.activate_num * np.log(self.depth):
            return False
        if (sum(self.feedback.values())/sum(self.vv.values()))<self.args.activate_prob * np.log(self.depth):
            return False
        return True

class LinUCBUserStruct():
    def __init__(self,uid,arms,dim,init,args):
        self.uid = uid
        self.arms = arms
        self.dim = dim
        self.A = np.identity(n=self.dim)
        self.Ainv = np.linalg.inv(self.A)
        self.b = np.zeros((self.dim))  
        if init!='zero':
            self.theta = np.random.rand(self.dim)
        else:
            self.theta = np.zeros(self.dim)
        self.alpha = args.gamma # store the new alpha calcuated in each iteratioin
        self.linucb = NaiveLinUCB(args)



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

class Item():
    def __init__(self,gid,x1):
        self.gid     = gid
        self.fv      = {} #feature vector for training/simulator
        self.fv['t'] = x1 #training

class pHCB(ContextualBanditLearner):
    def __init__(self,device, args,root, name='pHCB'):
        super(pHCB, self).__init__(args, device, name)
        
        self.device = device
        self.args = args
        self.name = name
        self.items = {}
        self.n_items = 0
        self.users={}
        self.user_pretrain_flag = []
        try:
            self.alpha=self.args.gamma
        except:
            self.alpha=-1
        self.root = root
        self.load_items()
        
    def load_items(self):
        nindex2embedding = np.load(os.path.join(self.args.root_data_dir, self.args.dataset,  'utils', 'nindex2embedding.npy'))
        for gid in range(len(nindex2embedding)):
            x1 = nindex2embedding[gid]
            self.items[gid] = Item(gid,x1)
        self.n_items=len(self.items)

    def sample_actions(self,uid):
        score_budget = self.per_rec_score_budget * self.rec_batch_size
        try:
            user=self.users[uid]
        except:
            arms = []
            root = self.root
            for _, node in root.children.items():
                arms.append(ArmStruct(node,2, self.args))
            dim = self.root.emb.shape[0]
            self.users[uid]=LinUCBUserStruct(uid,arms,dim,"zero",self.args)
            user=self.users[uid]
            
        
        if uid not in self.user_pretrain_flag and self.pretrained_mode and len(self.clicked_history[uid]) > 0:
            self.user_pretrain_flag.append(uid)
            for gid in self.clicked_history[uid]:
                for aid, arm in enumerate(user.arms):
                    if gid in arm.gids:
                        arm_picker = arm
                        item = self.items[gid]
                        picked_arms = [[item], arm_picker,aid]
                        self.update(None, picked_arms, [1], 'item', uid)
                        break

        
        
        if len(user.arms)<=score_budget:
            aids = list(range(len(user.arms)))
        else:
            aids = np.random.choice(len(user.arms),score_budget,replace=False)
            
        arms_emb = np.array([user.arms[index].arm.emb for index in aids])
        depths = np.array([user.arms[index].depth for index in aids])
        ucb = user.getProb(arms_emb) * depths
        aid_argmax = np.argsort(ucb)[::-1][:1].tolist()[0]
        arm_picker= user.arms[aid_argmax]
        # for index in aids:
        #     arm = user.arms[index]
        #     depth = arm.depth
        #     reward = user.getProb(arm.arm.emb)*depth
        #     if reward>max_r:
        #         aid = index
        #         max_r = reward
        # arm_picker = user.arms[aid]
        if len(arm_picker.gids)<=self.args.per_rec_score_budget:
            arms = [self.items[gid] for gid in arm_picker.gids]
        else:
            arms = [self.items[gid] for gid in np.random.choice(arm_picker.gids,self.args.per_rec_score_budget,replace=False).tolist()]
        
        items = self.users[user.uid].linucb.decide(user.uid,arms, self.rec_batch_size)
        return np.empty(0), (items, arm_picker, aid_argmax)


    def updateParameters(self, picked_arm, reward, uid):
        self.users[uid].updateParameters(picked_arm.arm.emb,reward)

    def update(self,topics,picked_arms,feedbacks, mode,uid = None):
        if mode == "item":
            items, arm_picker, aid = picked_arms[0], picked_arms[1], picked_arms[2]
            for item, feedback in zip(items, feedbacks):
            
                gid = item.gid
                arm_picker.feedback[gid] += feedback
                arm_picker.vv[gid] += 1
                arm_picker.itemclick[gid] = True
                user = self.users[uid]
                self.updateParameters(arm_picker,feedback,uid)
                if arm_picker.expand() and arm_picker.arm.is_leaf==False:
                    depth = arm_picker.depth+1
                    user.arms.pop(aid)
                    for _, node in arm_picker.arm.children.items():
                        arm = ArmStruct(node,depth, self.args)
                        for gid in arm.gids:
                            arm.itemclick[gid]=arm_picker.itemclick[gid]
                            arm.feedback[gid]=arm_picker.feedback[gid]
                            arm.vv[gid]=arm_picker.vv[gid]
                        user.arms.append(arm)
    
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