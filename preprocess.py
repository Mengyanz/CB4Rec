
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


if __name__ == "__main__":
    # from parameters import parse_args
    from thanhmachine_params import parse_args

    args = parse_args()
    news_preprocess(args)
    behavior_preprocess(args)