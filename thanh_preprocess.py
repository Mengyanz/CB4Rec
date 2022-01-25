
from collections import defaultdict,Counter
from tqdm import tqdm
import numpy as np
import random 
import re
import os
import pickle
from datetime import datetime 
import time 
date_format_str = '%m/%d/%Y %I:%M:%S %p'

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
        read_news(args, os.path.join(args.root_data_dir, "large/test/news.tsv"))

        for w, c in tqdm(word_cnt.items()):
            if c >= args.min_word_cnt:
                vocab_dict[w] = len(vocab_dict)

        news_index = np.zeros((len(news_info) + 1, args.max_title_len), dtype="float32")
        for nid in tqdm(nid2index):
            news_index[nid2index[nid]] = [
                vocab_dict[w] if w in vocab_dict else 0 for w in news_info[nid]
            ] + [0] * (args.max_title_len - len(news_info[nid]))

        glove_path = os.path.join(args.root_data_dir, "glove/glove.6B.300d.txt")
        embedding_matrix, exist_word = load_matrix(glove_path, vocab_dict)

        
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

def behavior_preprocess(args):
    out_path = os.path.join(args.root_data_dir, args.dataset, 'utils')
    tr_ctx_fname = os.path.join(out_path, "train_contexts.pkl")
    val_ctx_fname = os.path.join(out_path, "valid_contexts.pkl")

    # read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "train/behaviors.tsv"), 0, save=True)

    print('Preprocessing for Simulator ...') 
    # if os.path.exists(tr_ctx_fname):
    #     print('Loading from {}'.format(tr_ctx_fname))
    #     with open(tr_ctx_fname, 'rb') as f:
    #         tr_samples = pickle.load(f)
    # else:
    train_user_set, tr_samples, tr_sorted_samples, tr_user_indices = \
        read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "train/behaviors.tsv"), 0, save=True)

    # if os.path.exists(val_ctx_fname):
    #     print('Loading from {}'.format(val_ctx_fname))
    #     with open(val_ctx_fname, 'rb') as f:
    #         val_samples = pickle.load(f)
    # else:
    val_user_set, val_samples, val_sorted_samples, val_user_indices = \
        read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "valid/behaviors.tsv"), 1, save=True)

    print('Number of train users: {} (should be 711,222!)'.format(len(train_user_set)))

    print('Preprocessing for CB learner ...') 
    for trial in range(args.n_trials): 
        print('trial = {}'.format(trial))
        try:
            random_ids = np.load('./meta_data/indices_{}.npy'.format(trial))
        except:
            print('The meta data has not been generated.') 
            generate_random_ids_over_runs(args.n_trials) 
            time.sleep(5)
            random_ids = np.load('./meta_data/indices_{}.npy'.format(trial))
            # raise FileNotFoundError('You should run `generate_random_user_ids_over_runs` first!')

        print('Randomly select {} users from the train set'.format(args.num_selected_users)) 
        random_train_user_subset_ids = random_ids[:args.num_selected_users]
        random_user_subset = [train_user_set[i] for i in random_train_user_subset_ids]

        print('Saving the behaviour data of the selected users for the first split of the train data. ')
        cb_train_samples = [] 
        cb_valid_samples = []
        split_threshold = int(len(tr_sorted_samples) * args.cb_train_ratio) 
        print('Split threshold: {}/{}'.format(split_threshold,len(tr_sorted_samples)))
        
        selected_train_samples = [] 
        for i, sample in tqdm(enumerate(tr_sorted_samples)):
            uid = sample[3] 
            if uid in random_user_subset and i > split_threshold: # user in the selected set and it's recent samples. 
                cb_valid_samples.append(sample) 

            if uid not in random_user_subset and i <= split_threshold:
                cb_train_samples.append(sample)

        # Shuffle the list 
        random.shuffle(cb_train_samples)	
        random.shuffle(cb_valid_samples)	
        
        with open(os.path.join(out_path, "cb_train_contexts_nuser={}_splitratio={}_trial={}.pkl".format(args.num_selected_users, args.cb_train_ratio, trial)), "wb") as f:
            pickle.dump(cb_train_samples, f)
        with open(os.path.join(out_path, "cb_valid_contexts_nuser={}_splitratio={}_trial={}.pkl".format(args.num_selected_users, args.cb_train_ratio, trial)), "wb") as f:
            pickle.dump(cb_valid_samples, f)


def generate_random_ids_over_runs( num_trials = 10):
    # n_val_users = 255990
    n_train_users = 711222
    np.random.seed(2022)
    meta_data_path = './meta_data'
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


if __name__ == "__main__":
    # from parameters import parse_args
    from configs.thanh_params import parse_args

    args = parse_args()
    news_preprocess(args)
    behavior_preprocess(args)