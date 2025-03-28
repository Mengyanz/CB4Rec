from pathlib import Path
import time
import datetime
from tqdm import tqdm
from collections import defaultdict, Counter
import copy
import random
import re
import numpy as np
import os
import pickle

# config

name = 'large'

data_path = Path("/home/v-mezhang/blob/data/" + name)
out_path = Path("/home/v-mezhang/blob/data/" + name + "/utils_debug")
if not os.path.exists(out_path):
    os.mkdir(out_path)
glove_path = Path("/home/v-mezhang/blob/data/glove/glove.6B.300d.txt")

npratio = 4
max_his_len = 50
min_word_cnt = 5
max_title_len = 30


# news preprocess
news_info = {"<unk>": ""}
nid2index = {"<unk>": 0}
word_cnt = Counter()


def word_tokenize(sent):
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


for l in tqdm(open(data_path / "train" / "news.tsv", "r", encoding='utf-8')):
    nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
    if nid in nid2index:
        continue
    title = word_tokenize(title)[:max_title_len]
    nid2index[nid] = len(nid2index)
    news_info[nid] = title
    word_cnt.update(title)

for l in tqdm(open(data_path / "valid" / "news.tsv", "r", encoding='utf-8')):
    nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
    if nid in nid2index:
        continue
    title = word_tokenize(title)[:max_title_len]
    nid2index[nid] = len(nid2index)
    news_info[nid] = title
    word_cnt.update(title)

with open(out_path / "nid2index.pkl", "wb") as f:
    pickle.dump(nid2index, f)

with open(out_path / "news_info.pkl", "wb") as f:
    pickle.dump(news_info, f)

# if os.path.exists(data_path / "test" ):
#     test_news_info = {"<unk>": ""}
#     test_nid2index = {"<unk>": 0}
#     for l in tqdm(open(data_path / "test" / "news.tsv", "r", encoding='utf-8')):
#         nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
#         if nid in test_nid2index:
#             continue
#         title = word_tokenize(title)[:max_title_len]
#         test_nid2index[nid] = len(test_nid2index)
#         test_news_info[nid] = title
#         # word_cnt.update(title)

    # with open(out_path / "test_nid2index.pkl", "wb") as f:
    #     pickle.dump(test_nid2index, f)

    # with open(out_path / "test_news_info.pkl", "wb") as f:
    #     pickle.dump(test_news_info, f)

vocab_dict = {"<unk>": 0}

for w, c in tqdm(word_cnt.items()):
    if c >= min_word_cnt:
        vocab_dict[w] = len(vocab_dict)

with open(out_path / "vocab_dict.pkl", "wb") as f:
    pickle.dump(vocab_dict, f)

news_index = np.zeros((len(news_info) + 1, max_title_len), dtype="float32")

for nid in tqdm(nid2index):
    news_index[nid2index[nid]] = [
        vocab_dict[w] if w in vocab_dict else 0 for w in news_info[nid]
    ] + [0] * (max_title_len - len(news_info[nid]))

np.save(out_path / "news_index", news_index)

# if os.path.exists(data_path / "test" ):
#     test_news_index = np.zeros((len(test_news_info) + 1, max_title_len), dtype="float32")

#     for nid in tqdm(test_nid2index):
#         test_news_index[test_nid2index[nid]] = [
#             vocab_dict[w] if w in vocab_dict else 0 for w in test_news_info[nid]
#         ] + [0] * (max_title_len - len(test_news_info[nid]))

#     np.save(out_path / "test_news_index", test_news_index)


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

np.save(out_path / "embedding", embedding_matrix)

