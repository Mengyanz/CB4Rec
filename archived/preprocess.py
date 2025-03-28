
from collections import defaultdict,Counter
from tqdm import tqdm
import numpy as np
import re
import os
import pickle
from datetime import datetime 
date_format_str = '%m/%d/%Y %I:%M:%S %p'

news_info = {"<unk>": ""}
nid2index = {"<unk>": 0}
word_cnt = Counter()
vocab_dict = {"<unk>": 0}

user_imprs = defaultdict(list)

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
    out_path = os.path.join(args.root_data_dir, 'large/utils')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
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
    
def mezhang_news_preprocess(args):
    # news preprocess
    out_path = os.path.join(args.root_data_dir, 'large/utils')

    data_path = os.path.join(args.root_data_dir, 'large')
    glove_path = os.path.join(args.root_data_dir, "glove/glove.6B.300d.txt")
    npratio = 4
    max_his_len = 50
    min_word_cnt = 1
    max_title_len = 30

    news_info = {"<unk>": ""}
    nid2index = {"<unk>": 0}
    word_cnt = Counter()


    def word_tokenize(sent):
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        else:
            return []


    for l in tqdm(open(os.path.join(args.root_data_dir, "large/train/news.tsv"), "r", encoding='utf-8')):
        nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
        if nid in nid2index:
            continue
        title = word_tokenize(title)[:max_title_len]
        nid2index[nid] = len(nid2index)
        news_info[nid] = title
        word_cnt.update(title)

    for l in tqdm(open(os.path.join(args.root_data_dir, "large/valid/news.tsv"), "r", encoding='utf-8')):
        nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
        if nid in nid2index:
            continue
        title = word_tokenize(title)[:max_title_len]
        nid2index[nid] = len(nid2index)
        news_info[nid] = title
        word_cnt.update(title)

    with open(os.path.join(out_path, "nid2index.pkl"), "wb") as f:
        pickle.dump(nid2index, f)

    with open(os.path.join(out_path, "news_info.pkl"), "wb") as f:
        pickle.dump(news_info, f)

    if os.path.exists(os.path.join(data_path , "test") ):
        test_news_info = {"<unk>": ""}
        test_nid2index = {"<unk>": 0}
        for l in tqdm(open(os.path.join(args.root_data_dir, "large/test/news.tsv"), "r", encoding='utf-8')):
            nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
            if nid in test_nid2index:
                continue
            title = word_tokenize(title)[:max_title_len]
            test_nid2index[nid] = len(test_nid2index)
            test_news_info[nid] = title
            # word_cnt.update(title)

        with open(os.path.join(out_path, "test_nid2index.pkl"), "wb") as f:
            pickle.dump(test_nid2index, f)

        with open(os.path.join(out_path, "test_news_info.pkl"), "wb") as f:
            pickle.dump(test_news_info, f)

    vocab_dict = {"<unk>": 0}

    for w, c in tqdm(word_cnt.items()):
        if c >= min_word_cnt:
            vocab_dict[w] = len(vocab_dict)

    with open(os.path.join(out_path, "vocab_dict.pkl"), "wb") as f:
        pickle.dump(vocab_dict, f)

    news_index = np.zeros((len(news_info) + 1, max_title_len), dtype="float32")

    for nid in tqdm(nid2index):
        news_index[nid2index[nid]] = [
            vocab_dict[w] if w in vocab_dict else 0 for w in news_info[nid]
        ] + [0] * (max_title_len - len(news_info[nid]))

    np.save(os.path.join(out_path, "news_index"), news_index)

    if os.path.exists(os.path.join(data_path, "test") ):
        test_news_index = np.zeros((len(test_news_info) + 1, max_title_len), dtype="float32")

        for nid in tqdm(test_nid2index):
            test_news_index[test_nid2index[nid]] = [
                vocab_dict[w] if w in vocab_dict else 0 for w in test_news_info[nid]
            ] + [0] * (max_title_len - len(test_news_info[nid]))

        np.save(os.path.join(out_path, "test_news_index"), test_news_index)


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


    embedding_matrix, exist_word = load_matrix(glove_path, vocab_dict)

    print(embedding_matrix.shape[0], len(exist_word))

    np.save(os.path.join(out_path , "embedding"), embedding_matrix)

def read_imprs(args, path, mode):
    index = 0
    samples = []
    user_indices = defaultdict(list)
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
    

    if mode == 0:
        name = 'train'
    elif mode == 1:
        name = 'valid'
    else:
        name = 'test'

    with open(os.path.join(out_path, (name + "_sam_uid.pkl")), "wb") as f:
        pickle.dump(samples, f)
    with open(os.path.join(out_path, (name + "_user_indices.pkl")), "wb") as f:
        pickle.dump(user_indices, f)
    with open(os.path.join(out_path, ("sorted_"+ name + "_sam_uid.pkl")), "wb") as f:
        pickle.dump(sorted_samples, f)

def behavior_preprocess(args):
    read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "train/behaviors.tsv"), 0)
    read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "valid/behaviors.tsv"), 1)
    if os.path.exists(os.path.join(args.root_data_dir, args.dataset, "test/behaviors.tsv")):
        read_imprs(args, os.path.join(args.root_data_dir, args.dataset, "test/behaviors.tsv"), 2)


def generate_cb_users(args):
    """
    randomly sample users from valid data for contextual bandits simulation. 
    generate cb_users: list of user samples ([pos, neg_imp, his, uid, tsp])
    save to file cb_users.npy
    """
    data_path = "/home/v-mezhang/blob/data/large/utils/valid_sam_uid.pkl"
    valid_user = {}

    with open(os.path.join(args.root_data_dir, args.dataset, 'utils/valid_sam_uid.pkl'), 'rb') as f:
        valid_sam = pickle.load(f)
        for sam in valid_sam:
            uid = sam[3]
            # record the first time a user appears 
            # we will only use uid, his in cb simulation anyway
            if uid not in valid_user:
                valid_user[uid] = sam 
                
    uids = np.random.choice(list(valid_user.keys()), size = args.num_users, replace = False)
    cb_users = []
    for uid in uids:
        cb_users.append(sam)
    # print(cb_users)
    np.save("/home/v-mezhang/blob/data/large/cb_users", cb_users)
        

def generate_cb_news(args):
    """
    Generate candidate news who subcat having #news>= 200 for cb simulation.
    generate cb_news: dict, key: subvert; value: list of news samples
    save to file cb_newss.npy
    """
    data_path = "/home/v-mezhang/blob/data/large/train_valid/news.tsv"
    cat_count = {}
    subcat_count = {}
    news_dict = {}

    for l in tqdm(open(data_path, "r", encoding='utf-8')):
        nid, vert, subvert, _, _, _, _, _ = l.strip("\n").split("\t")
        if nid not in news_dict:
            news_dict[nid] = l
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
        if subcat_count[subvert] >= 200:
            cb_news[subvert].append(l)
            
    # np.save("/home/v-mezhang/blob/data/large/cb_news", cb_news)
    with open("/home/v-mezhang/blob/data/large/cb_news.pkl", "wb") as f:
        pickle.dump(cb_news, f)


if __name__ == "__main__":
    from CB4Rec.configs.params import parse_args
    # from thanh_params import parse_args

    args = parse_args()
    # news_preprocess(args)
    # behavior_preprocess(args)
    # generate_cb_users(args)
    generate_cb_news(args)