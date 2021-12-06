
from collections import Counter
from tqdm import tqdm
import numpy as np
import re
import os
import pickle

news_info = {"<unk>": ""}
nid2index = {"<unk>": 0}
word_cnt = Counter()
vocab_dict = {"<unk>": 0}

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
    out_path = os.path.join(args.root_data_dir, 'utils')
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

    np.save(os.path.join(out_path,"embedding"), embedding_matrix)

def behavior_preprocess(args):
    pass


if __name__ == "__main__":
    from parameters import parse_args
    args = parse_args()
    news_preprocess(args)