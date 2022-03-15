import pickle, os
from collections import defaultdict
import argparse
import logging
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
        
import pickle

parser = argparse.ArgumentParser()

# path
parser.add_argument("--root_data_dir",type=str,default="/home/featurize/CB4Rec/data/")
parser.add_argument("--root_proj_dir",type=str,default="/home/featurize/CB4Rec/")
# parser.add_argument("--root_proj_dir",type=str,default="./")
# parser.add_argument("--model_path", type=str, default="/home/v-zhenyuhe/CB4Rec/model/large/large.pkl")
parser.add_argument("--sim_path", type=str, default="pretrained_models/sim_nrms_bce_r14_ep6_thres038414")
parser.add_argument("--sim_threshold", type=float, default=0.38414)

# Preprocessing 
parser.add_argument("--dataset",type=str,default='large')
parser.add_argument("--num_selected_users", type=int, default=1000, help='number of randomly selected users from val set')
parser.add_argument("--cb_train_ratio", type=float, default=0.2)
parser.add_argument("--sim_npratio", type=int, default=4)
parser.add_argument("--sim_val_batch_size", type=int, default=1024)

# Simulation
parser.add_argument("--algo",type=str,default="ts_neuralucb")
parser.add_argument("--algo_prefix", type=str, default="algo",
    help='the name of save files')
parser.add_argument("--n_trials", type=int, default=4, help = 'number of experiment runs')
parser.add_argument("--T", type=int, default=1000, help = 'number of rounds (interactions)')
parser.add_argument("--topic_update_period", type=int, default=1, help = 'Update period for CB topic model')
parser.add_argument("--update_period", type=int, default=100, help = 'Update period for CB model')
parser.add_argument("--n_inference", type=int, default=5, help='number of Monte Carlo samples of prediction. ')
parser.add_argument("--rec_batch_size", type=int, default=5, help='recommendation size for each round.')
parser.add_argument("--per_rec_score_budget", type=int, default=1000, help='buget for calcuating scores, e.g. ucb, for each rec')
parser.add_argument("--max_batch_size", type=int, default=256, help = 'Maximum batch size your GPU can fit in.')
parser.add_argument("--pretrained_mode",type=bool,default=True, 
    help="Indicates whether to load a pretrained model. True: load from a pretrained model, False: no pretrained model ")
parser.add_argument("--preinference_mode",type=bool,default=True, 
    help="Indicates whether to preinference news before each model update.")

parser.add_argument("--uniform_init",type=bool,default=True, 
    help="For Thompson Sampling: Indicates whether to init ts parameters uniformly")
parser.add_argument("--gamma", type=float, default=1.0, help='ucb parameter: mean + gamma * std.')


# nrms 
parser.add_argument("--npratio", type=int, default=4) 
parser.add_argument("--max_his_len", type=int, default=50)
parser.add_argument("--min_word_cnt", type=int, default=1) # 5
parser.add_argument("--max_title_len", type=int, default=30)
# nrms topic
parser.add_argument("--dynamic_aggregate_topic", type=bool, default=True) # whether to dynamicly aggregate small topic during simulation
parser.add_argument("--min_item_size", type=int, default=1000)

# model training
parser.add_argument("--batch_size", type=int, default=64) 
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

class Tree():
    def __init__(self):
        self.emb         = None
        self.size        = 0
        self.item_index  = []
        self.gids = []
        self.children    = None
        self.is_leaf     = False

# root_data_dir = "/home/featurize/CB4Rec/data"
# dataset = "large"
# with open(os.path.join("/home/featurize/CB4Rec/data", "large",  'utils', 'nid2index.pkl'), 'rb') as f:
#     nid2index = pickle.load(f)
# news_index = np.load(os.path.join(root_data_dir, dataset,  'utils', 'news_index.npy'))

# nindex2embedding = np.load(os.path.join(args.root_data_dir, args.dataset,  'utils', 'nindex2embedding.npy'))

# items_cluster = {}

# num_cluster = 10
# kmeans = KMeans(n_clusters=num_cluster, random_state=2022).fit(nindex2embedding)
# items_cluster['layer1'] = kmeans.labels_.tolist()

# c = Counter(items_cluster['layer1'])


# root = Tree()
# root.size = len(nid2index)
# root.emb = np.mean(nindex2embedding[:-1], axis=0)
# root.item_index = range(len(nid2index))


# from tqdm import tqdm
# root.children = {}
# for index, item_index in enumerate(root.item_index):
#     cluster = items_cluster['layer1'][index]
#     if cluster not in root.children:
#         root.children[cluster] = Tree()
#         root.children[cluster].size = dict(c)[cluster]
#     else:
#         root.children[cluster].item_index.append(item_index)
# for cluster in root.children.keys():
#     root.children[cluster].emb = np.mean(nindex2embedding[root.children[cluster].item_index], axis=0)
    

# for cluster in tqdm(root.children.keys(), total=10):
#     root.children[cluster].children = {}
#     num_cluster = 100
#     kmeans = KMeans(n_clusters=num_cluster, random_state=2022).fit(nindex2embedding)
#     cluster_labels = kmeans.labels_.tolist()
#     c = Counter(cluster_labels)
#     for index, item_index in enumerate(root.children[cluster].item_index):
#         cluster_2 = cluster_labels[index]
#         if cluster_2 not in root.children[cluster].children:
#             root.children[cluster].children[cluster_2] = Tree()
#             root.children[cluster].children[cluster_2].size = dict(c)[cluster_2]
#         else:
#             root.children[cluster].children[cluster_2].item_index.append(item_index)
# for cluster in root.children.keys():  
#     for cluster_2 in root.children[cluster].children.keys():
#         root.children[cluster].children[cluster_2].emb = np.mean(nindex2embedding[root.children[cluster].children[cluster_2].item_index], axis=0)
        
        
with open(os.path.join(args.root_data_dir, 'my_tree.pkl'), 'rb') as f:
    root = pickle.load(f)

for cluster in root.children.keys():
    root.children[cluster].gids = root.children[cluster].item_index
    for cluster_2 in root.children[cluster].children.keys():
        root.children[cluster].children[cluster_2].is_leaf=True
        root.children[cluster].children[cluster_2].gids = root.children[cluster].children[cluster_2].item_index
        
with open(os.path.join(args.root_data_dir, 'my_tree.pkl'), 'wb') as f:
    pickle.dump(root, f)
    
    
    
    
