
from collections import defaultdict,Counter
from tqdm import tqdm
import numpy as np
import random 
import re
import os
import pickle
import json
from datetime import datetime 
import time 
date_format_str = '%m/%d/%Y %I:%M:%S %p'

from algorithms.nrms_model import NRMS_Model, NRMS_Topic_Model
import torch
from torch.utils.data import DataLoader
from utils.data_util import TrainDataset, NewsDataset, UserDataset
from torch import nn
import torch.optim as optim
from metrics import evaluation_split

os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
device = torch.device("cuda:0")
torch.cuda.set_device(device)

news_info = {"<unk>": ""}
nid2index = {"<unk>": 0}
word_cnt = Counter()
vocab_dict = {"<unk>": 0}

# user_imprs = defaultdict(list)

def word_tokenize(sent):
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def load_matrix(glove_path, word_dict):
    # embebbed_dict = {}
    embedding_matrix = np.zeros((len(word_dict) + 1, 300))
    exist_word = []

    # get embedded_dict
    with open(glove_path, "rb") as f:
        for l in tqdm(f):
            l = l.split()
            word = l[0].decode()
            if len(word) != 0 and word in word_dict:
                wordvec = [float(x) for x in l[1:]]
                index = word_dict[word]
                embedding_matrix[index] = np.array(wordvec)
                exist_word.append(word)

    # get union
    return embedding_matrix, exist_word

def read_news(args, path):
    for l in tqdm(open(path, "r", encoding='utf-8')):
        nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
        if nid in nid2index:
            continue
        title = word_tokenize(title)[:args.max_title_len]
        nid2index[nid] = len(nid2index)
        news_info[nid] = title
        word_cnt.update(title)

def news_preprocess(args):
    """
    Output:
        news_index: dict, 
            key: a news id 
            value: a vector representation for a news, vector length = args.max_title_len 
    """
    print('Converting to word embedding using glove6B!') 

    out_path = os.path.join(args.root_data_dir, 'large/utils')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    fname = os.path.join(out_path,"nid2index.pkl")
    if os.path.exists(fname):
        print('The word embedding has been generated in {}. No need to do anything here!'.format(out_path))

    else:
        read_news(args, os.path.join(args.root_data_dir, "large/train/news.tsv"))
        read_news(args, os.path.join(args.root_data_dir, "large/valid/news.tsv"))
        # read_news(args, os.path.join(args.root_data_dir, "large/test/news.tsv"))

        for w, c in tqdm(word_cnt.items()):
            if c >= args.min_word_cnt:
                vocab_dict[w] = len(vocab_dict)

        news_index = np.zeros((len(news_info) + 1, args.max_title_len), dtype="float32") # vect representation for each news
        for nid in tqdm(nid2index):
            news_index[nid2index[nid]] = [
                vocab_dict[w] if w in vocab_dict else 0 for w in news_info[nid]
            ] + [0] * (args.max_title_len - len(news_info[nid]))

        glove_path = os.path.join(args.root_data_dir, "glove/glove.6B.300d.txt")
        embedding_matrix, exist_word = load_matrix(glove_path, vocab_dict)
        print('Debug embedding matrix shape: ', embedding_matrix.shape)

        
        with open(os.path.join(out_path,"nid2index.pkl"), "wb") as f:
            pickle.dump(nid2index, f)

        with open(os.path.join(out_path,"news_info.pkl"), "wb") as f:
            pickle.dump(news_info, f)

        with open(os.path.join(out_path,"vocab_dict.pkl"), "wb") as f:
            pickle.dump(vocab_dict, f)

        np.save(os.path.join(out_path,"news_index"), news_index)
        np.save(os.path.join(out_path,"embedding"), embedding_matrix)

def read_imprs(args, path, mode, save=False):
    """
    Args:
        mode: 0 (train), 1 (valid)
    """
    index = 0
    samples = []
    user_indices = defaultdict(list)
    user_imprs = defaultdict(list)
    out_path = os.path.join(args.root_data_dir, args.dataset, 'utils')

    for l in tqdm(open(path, "r")):
        imp_id, uid, t, his, imprs = l.strip("\n").split("\t")

        his = his.split()
        tsp = t
        # tsp = time.mktime(time.strptime(t, "%m/%d/%Y %I:%M:%S %p"))
        #tsp = int(t)
        imprs = [i.split("-") for i in imprs.split(" ")]
        neg_imp = [i[0] for i in imprs if i[1] == "0"]
        pos_imp = [i[0] for i in imprs if i[1] == "1"]
        user_imprs[uid].append([tsp, his, pos_imp, neg_imp, mode, uid])

        his = his[-args.max_his_len:]
        if mode == 0:
            for pos in pos_imp:
                samples.append([pos, neg_imp, his, uid, tsp])
                user_indices[uid].append(index)
                index += 1
        else:
            samples.append([pos_imp, neg_imp, his, uid, tsp])

    sorted_samples = [i for i in sorted(samples, key=lambda date: datetime.strptime(date[-1], date_format_str))]

    name = 'train' if mode == 0 else 'valid'
    if save: 
        with open(os.path.join(out_path, (name + "_contexts.pkl")), "wb") as f:
            pickle.dump(samples, f)
        with open(os.path.join(out_path, (name + "_user_indices.pkl")), "wb") as f:
            pickle.dump(user_indices, f)
        with open(os.path.join(out_path, ("sorted_"+ name + "_contexts.pkl")), "wb") as f:
            pickle.dump(sorted_samples, f)

    user_set = list(user_imprs)
    return user_set, samples, sorted_samples, user_indices

# def generate_random_ids_over_runs( num_trials = 10):
def generate_random_ids_over_runs(num_trials, meta_data_path):
# >>>>>>> d58630b1cc4f37dcdcbc90e55e93d45e49308639
    # n_val_users = 255990
    n_train_users = 711222
    np.random.seed(2022)
    print('WARNING: This is to generate meta data for dataset generation, and should only be performed once.' 
        'Quit now if you are not sure what you are doing!!!')
    s = input('Type yesimnotstupid to proceed: ')
    if s == 'yesimnotstupid':
        if not os.path.exists(meta_data_path):
            os.mkdir(meta_data_path) 

        for sim_id in range(num_trials):
            np.random.seed(sim_id)
            indices = np.random.permutation(n_train_users)
            np.save(os.path.join(meta_data_path, 'indices_{}.npy'.format(sim_id)), indices)

def read_imprs_for_val_set_for_sim(args, path):
    """
    Args:
        mode: 0 (train), 1 (valid)
    """
    samples = []
    out_path = os.path.join(args.root_data_dir, args.dataset, 'utils')

    for l in tqdm(open(path, "r")):
        imp_id, uid, t, his, imprs = l.strip("\n").split("\t")

        his = his.split()
        tsp = t
        # tsp = time.mktime(time.strptime(t, "%m/%d/%Y %I:%M:%S %p"))
        #tsp = int(t)
        imprs = [i.split("-") for i in imprs.split(" ")]
        neg_imp = [i[0] for i in imprs if i[1] == "0"]
        pos_imp = [i[0] for i in imprs if i[1] == "1"]

        his = his[-args.max_his_len:]
        labels = [1] * len(pos_imp) + [0] * len(neg_imp) 
        nns = pos_imp + neg_imp 
        samples.append([nns, labels, his, uid, tsp])
        # for n,l in zip(nns, labels):
            # samples.append([n, l, his, uid, tsp]) 

    with open(os.path.join(out_path, "val_contexts.pkl"), "wb") as f:
        pickle.dump(samples, f)


def behavior_preprocess(args):
    out_path = os.path.join(args.root_data_dir, args.dataset, 'utils')
    tr_ctx_fname = os.path.join(out_path, "train_contexts.pkl")
    val_ctx_fname = os.path.join(out_path, "valid_contexts.pkl")

    # read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "train/behaviors.tsv"), 0, save=True)

    print('Preprocessing for Simulator ...') 
    if os.path.exists(tr_ctx_fname):
        print('{} is already created!'.format(tr_ctx_fname))
    else:
        read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "train/behaviors.tsv"), 0, save=True)

    if os.path.exists(val_ctx_fname):
        print('{} is already created!'.format(val_ctx_fname))
    else:
        read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "valid/behaviors.tsv"), 1, save=True)

    train_user_set, _, tr_rep_sorted_samples, _ = \
        read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "train/behaviors.tsv"), 1) 

    print('Number of train users: {} (should be 711,222!)'.format(len(train_user_set)))

    # Create a click history for each user in train: 
    # Each user in the MIND train has the same clicked history across samples
    # @TODO: Consider updating the clicked history of each user at different times in the MIND train, to train the simulator. 
    # so that the simulator has a larger clicked history than CB learner. Or just don't update it because the impression list is already larger? 
    # Note that:
    #   * To train a simulator is to train its news and user encoders - it uses both the clicked history and impression set 
    #   * To run (or evaluate) a simulator is to run its trained news and user encoders - only clicked history is required 
    #   * The same comments apply to a CB learner
    clicked_history = defaultdict(list)
    for sample in tqdm(tr_rep_sorted_samples): 
            uid = sample[3]
            if uid not in clicked_history: 
                clicked_history[uid] = sample[2] 
        
    with open(os.path.join(out_path, "train_clicked_history.pkl"), "wb") as f:
        pickle.dump(clicked_history, f)

#     print('Preprocessing for CB learner ...') 

#     for trial in range(args.n_trials): 
#         print('trial = {}'.format(trial))

#         cb_train_fname = os.path.join(out_path, "cb_train_contexts_nuser={}_splitratio={}_trial={}.pkl".format(args.num_selected_users, args.cb_train_ratio, trial))
#         cb_valid_fname = os.path.join(out_path, "cb_valid_contexts_nuser={}_splitratio={}_trial={}.pkl".format(args.num_selected_users, args.cb_train_ratio, trial))

#         if os.path.exists(cb_train_fname):
#             continue


# #         meta_data_path = os.path.join(args.root_data_dir, args.dataset, 'meta_data')

#         try:
#             random_ids = np.load(os.path.join(meta_data_path, 'indices_{}.npy'.format(trial)))
#         except:
#             print('The meta data has not been generated.') 
#             generate_random_ids_over_runs(args.n_trials, meta_data_path) 
#             time.sleep(5)
#             random_ids = np.load(os.path.join(meta_data_path, 'indices_{}.npy'.format(trial)))
#             # raise FileNotFoundError('You should run `generate_random_user_ids_over_runs` first!')

#         print('Randomly select {} users from the train set'.format(args.num_selected_users)) 
#         random_train_user_subset_ids = random_ids[:args.num_selected_users]
#         random_user_subset = [train_user_set[i] for i in random_train_user_subset_ids]

#         print('Saving the behaviour data of the selected users for the first split of the train data. ')
#         cb_train_samples = [] 
#         cb_valid_samples = []
#         split_threshold = int(len(tr_rep_sorted_samples) * args.cb_train_ratio) 
#         print('Split threshold: {}/{}'.format(split_threshold,len(tr_rep_sorted_samples)))
        
#         selected_train_samples = [] 
#         for i, sample in tqdm(enumerate(tr_rep_sorted_samples)):
#             uid = sample[3] 
#             if uid in random_user_subset and i > split_threshold: # user in the selected set and it's recent samples. 
#                 cb_valid_samples.append(sample) 

#             if uid not in random_user_subset and i <= split_threshold:
#                 pos_imp, neg_imp, his, uid, tsp = sample
#                 for pos in pos_imp:
#                     cb_train_samples.append([pos, neg_imp, his, uid, tsp])


#         # Shuffle the list 
#         random.shuffle(cb_train_samples)	
#         # random.shuffle(cb_valid_samples)	
        
#         with open(cb_train_fname, "wb") as f:
#             pickle.dump(cb_train_samples, f)
#         with open(cb_valid_fname, "wb") as f:
#             pickle.dump(cb_valid_samples, f)


def split_then_select_behavior_preprocess(args):
    out_path = os.path.join(args.root_data_dir, args.dataset, 'utils')
    tr_ctx_fname = os.path.join(out_path, "train_contexts.pkl")
    val_ctx_fname = os.path.join(out_path, "valid_contexts.pkl")

    # read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "train/behaviors.tsv"), 0, save=True)

    print('Preprocessing for Simulator ...') 
    if os.path.exists(tr_ctx_fname):
        print('{} is already created!'.format(tr_ctx_fname))
    else:
        read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "train/behaviors.tsv"), 0, save=True)

    if os.path.exists(val_ctx_fname):
        print('{} is already created!'.format(val_ctx_fname))
    else:
        read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "valid/behaviors.tsv"), 1, save=True)

    # train_user_set, _, tr_rep_sorted_samples, _ = \
    #     read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "train/behaviors.tsv"), 0) 

    # print('Number of train users: {} (should be 711,222!)'.format(len(train_user_set)))
    print('Loading sorted_train_contexts.pkl ...')
    with open(os.path.join(out_path, ("sorted_train_contexts.pkl")), "rb") as f:
        tr_rep_sorted_samples = pickle.load(f)

    # Create a click history for each user in train: 
    # Each user in the MIND train has the same clicked history across samples
    # @TODO: Consider updating the clicked history of each user at different times in the MIND train, to train the simulator. 
    # so that the simulator has a larger clicked history than CB learner. Or just don't update it because the impression list is already larger? 
    # Note that:
    #   * To train a simulator is to train its news and user encoders - it uses both the clicked history and impression set 
    #   * To run (or evaluate) a simulator is to run its trained news and user encoders - only clicked history is required 
    #   * The same comments apply to a CB learner
    clicked_history = defaultdict(list)
    for sample in tqdm(tr_rep_sorted_samples): 
            uid = sample[3]
            if uid not in clicked_history: 
                clicked_history[uid] = sample[2] 
  
    with open(os.path.join(out_path,"nid2index.pkl"), "rb") as f:
        nid2index = pickle.load(f)
    
    for u,v in clicked_history.items(): 
        clicked_history[u] = [nid2index[l] for l in v]
        
    with open(os.path.join(out_path, "train_clicked_history.pkl"), "wb") as f:
        pickle.dump(clicked_history, f)

    print('Preprocessing for CB learner ...') 

    # Split the MIND train 
    # if not os.path.exists(os.path.join(out_path, 'cb_val_users.pkl')):
    split_threshold = int(len(tr_rep_sorted_samples) * args.cb_train_ratio) 
    print('Split threshold: {}/{}'.format(split_threshold,len(tr_rep_sorted_samples)))
    cb_train = [] 
    cb_val = []
    cb_val_users = []
    for i, sample in tqdm(enumerate(tr_rep_sorted_samples)):
        uid = sample[3]
        if i > split_threshold: # user in the selected set and it's recent samples. 
            cb_val.append(sample) 
            cb_val_users.append(uid)
        else:
            cb_train.append(sample)

    cb_val_users = list(set(cb_val_users))

    print('#cb_val_users: {}'.format(len(cb_val_users)))
    with open(os.path.join(out_path, 'cb_val_users.pkl'), 'wb') as fo: 
        pickle.dump(cb_val_users, fo) 
    # else:
    #     with open(os.path.join(out_path, 'cb_val_users.pkl'), 'rb') as f: 
    #         cb_val_users = pickle.load(f) 
    #     print('Load cb_val_users -- #cb_val_users: {}'.format(len(cb_val_users)))

    # meta_data_path = './meta_data'
    meta_data_path = os.path.join(args.root_proj_dir, 'meta_data')
    if not os.path.exists(meta_data_path):
        os.mkdir(meta_data_path) 
    for trial in range(args.n_trials): 
        np.random.seed(trial)
        indices_path = os.path.join(meta_data_path, 'indices_{}.npy'.format(trial))
        if not os.path.exists(indices_path):
            indices = np.random.permutation(len(cb_val_users))
            np.save(indices_path, indices)

            cb_train_fname = os.path.join(out_path, "cb_train_contexts_nuser={}_splitratio={}_trial={}.pkl".format(args.num_selected_users, args.cb_train_ratio, trial))
            # cb_valid_fname = os.path.join(out_path, "cb_valid_contexts_nuser={}_splitratio={}_trial={}.pkl".format(args.num_selected_users, args.cb_train_ratio, trial))

            rand_user_set  = [cb_val_users[i] for i in indices[:args.num_selected_users] ]
            cb_train_uremoved = []
            for sample in cb_train: 
                uid = sample[3] 
                if uid not in rand_user_set: 
                    cb_train_uremoved.append(sample)

            # np.random.shuffle(cb_train_uremoved)
            with open(cb_train_fname, "wb") as f:
                pickle.dump(cb_train_uremoved, f)
            # with open(cb_valid_fname, "wb") as f:
            #    pickle.dump(cb_val, f)

            pretrain_cb_learner(args, cb_train_uremoved, trial)
        else:
            print('{} exists!'.format(indices_path))
    

def pretrain_cb_learner(args, cb_train_sam, trial):
    """pretrain cb learner based on each trial's cb_train 
    Args:
        cb_train_sam: list of samples for cb learner
        trial: int, trial number
    Return 
        trained cb learner (nrms) and save to file
    """
    
    # out path
    if args.pretrain_topic:
        out_model_path = os.path.join(args.root_proj_dir, 'cb_topic_pretrained_models')
    else:
        out_model_path = os.path.join(args.root_proj_dir, 'cb_pretrained_models')
    if not os.path.exists(out_model_path):
        os.mkdir(out_model_path)
    log_path = os.path.join(args.root_proj_dir, 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path) 

    # load data
    if args.pretrain_topic:
        with open(os.path.join(args.root_data_dir, args.dataset,  'utils', 'nid2topicindex.pkl'), 'rb') as f:
            nid2topicindex = pickle.load(f)
            
    with open(os.path.join(args.root_data_dir, args.dataset,  'utils', 'nid2index.pkl'), 'rb') as f:
        nid2index = pickle.load(f)
    word2vec = np.load(os.path.join(args.root_data_dir, args.dataset,  'utils', 'embedding.npy'))
    nindex2vec = np.load(os.path.join(args.root_data_dir, args.dataset,  'utils', 'news_index.npy'))

    # REVIEW: randomly selecting 90% of the data for training and 10% for validation
    train_idx = np.random.choice(range(len(cb_train_sam)), size = int(0.9 * len(cb_train_sam)), replace=False)
    valid_idx = list(set(range(len(cb_train_sam))) - set(train_idx))
    train_sam = [cb_train_sam[i] for i in train_idx]
    valid_sam = [cb_train_sam[i] for i in valid_idx]

    if args.pretrain_topic:
        train_ds = TrainDataset(args, train_sam, nid2index,  nindex2vec, nid2topicindex)
    else:
        train_ds = TrainDataset(args, train_sam, nid2index,  nindex2vec)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.pretrain_topic:
        model = NRMS_Topic_Model(word2vec).to(device)
    else:
        model = NRMS_Model(word2vec).to(device)
    
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

        if args.pretrain_topic:
            val_scores = eva(args, model, valid_sam, nid2index,  nindex2vec, nid2topicindex)
        else:
            val_scores = eva(args, model, valid_sam, nid2index,  nindex2vec)  
        val_auc, val_mrr, val_ndcg, val_ndcg10, ctr = [np.mean(i) for i in list(zip(*val_scores))]
        print(f"[{ep}] epoch auc: {val_auc:.4f}, mrr: {val_mrr:.4f}, ndcg5: {val_ndcg:.4f}, ndcg10: {val_ndcg10:.4f}, ctr: {ctr:.4f}")

        with open(os.path.join(log_path, 'indices_{}.txt'.format(trial)), 'a') as f:
            f.write(f"[{ep}] epoch auc: {val_auc:.4f}, mrr: {val_mrr:.4f}, ndcg5: {val_ndcg:.4f}, ndcg10: {val_ndcg10:.4f} , ctr: {ctr:.4f}\n")  
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(out_model_path, 'indices_{}.pkl'.format(trial)))
            with open(os.path.join(log_path, 'indices_{}.txt'.format(trial)), 'a') as f:
                f.write(f"[{ep}] epoch save model\n")

def eva(args, model, valid_sam, nid2index, news_index, nid2topicindex=None):
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
        if args.pretrain_topic:
            user_vec = model.dimmension_reduction(model.user_encoder(his)).detach().cpu().numpy()
        else:
            user_vec = model.user_encoder(his).detach().cpu().numpy()
        user_vecs.append(user_vec)
    user_vecs = np.concatenate(user_vecs)
    
    if args.pretrain_topic:
        topic_vecs = model.get_all_topic_embedding().detach().cpu().numpy()
        val_scores = evaluation_split(news_vecs, user_vecs, valid_sam, nid2index, nid2topicindex=nid2topicindex, topic_vecs=topic_vecs)
    else:
        val_scores = evaluation_split(news_vecs, user_vecs, valid_sam, nid2index)
    
    return val_scores


def generate_cb_news(args):
    """
    Generate candidate news who subcat having #news>= 200 (@Thanh: I removed it here) for cb simulation.
    generate cb_news: dict, key: subvert; value: list of news samples
    save to file cb_news.pkl
    """
    # data_path = "/home/v-mezhang/blob/data/large/train_valid/news.tsv"
    cat_count = {}
    subcat_count = {}
    news_dict = {}
    nid2topic = {}
    nid2topicindex = {}
    topic_ordered_list = json.load(open(os.path.join(args.root_data_dir, "large/utils/subcategory_byorder.json"), 'r'))
    topic2index = {}
    for topic in topic_ordered_list:
        topic2index[topic] = len(topic2index)
        
    train_news_path = os.path.join(args.root_data_dir, "large/train/news.tsv") 
    valid_news_path = os.path.join(args.root_data_dir, "large/valid/news.tsv")
    news_paths = [train_news_path, valid_news_path]

    for data_path in news_paths:
        for l in tqdm(open(data_path, "r", encoding='utf-8')):
            nid, vert, subvert, _, _, _, _, _ = l.strip("\n").split("\t")
            if nid not in news_dict:
                news_dict[nid] = l
                nid2topic[nid] = subvert
                nid2topicindex[nid] = topic2index[subvert]
                if vert not in cat_count:
                    cat_count[vert] = 1
                else:
                    cat_count[vert] += 1
                if subvert not in subcat_count:
                    subcat_count[subvert] = 1
                else:
                    subcat_count[subvert] += 1

    cb_news = defaultdict(list)
    for nid, l in news_dict.items():
        subvert = l.strip("\n").split("\t")[2]
        # if subcat_count[subvert] >= 200:
        cb_news[subvert].append(l)
            
    # np.save("/home/v-mezhang/blob/data/large/cb_news", cb_news)
    save_path = os.path.join(args.root_data_dir, "large/utils/cb_news.pkl") 
    with open(save_path, "wb") as f:
        pickle.dump(cb_news, f)
        
    save_path = os.path.join(args.root_data_dir, "large/utils/nid2topic.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(nid2topic, f)
    save_path = os.path.join(args.root_data_dir, "large/utils/nid2topicindex.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(nid2topicindex, f)

if __name__ == "__main__":
    # from parameters import parse_args
    # from configs.thanh_params import parse_args
    # from configs.mezhang_params import parse_args
    from configs.zhenyu_params import parse_args


    args = parse_args()
    news_preprocess(args)
    # read_imprs_for_val_set_for_sim(args, path)
    generate_cb_news(args)
    behavior_preprocess(args)
    split_then_select_behavior_preprocess(args)

    # Get val set for sim 
    read_imprs_for_val_set_for_sim(args, os.path.join(args.root_data_dir, args.dataset, "valid/behaviors.tsv"))
