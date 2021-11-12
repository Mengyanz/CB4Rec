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
data_path = Path("../blob/data/demo")
out_path = Path("../blob/data/demo/utils")

npratio = 4
max_his_len = 50
min_word_cnt = 3
max_title_len = 30

user_imprs = defaultdict(list)

# read user impressions
for l in tqdm(open(data_path / "train" / "behaviors.tsv", "r")):
    imp_id, uid, t, his, imprs = l.strip("\n").split("\t")
    his = his.split()
    tsp = t
    # tsp = time.mktime(time.strptime(t, "%m/%d/%Y %I:%M:%S %p"))
    #tsp = int(t)
    imprs = [i.split("-") for i in imprs.split(" ")]
    neg_imp = [i[0] for i in imprs if i[1] == "0"]
    pos_imp = [i[0] for i in imprs if i[1] == "1"]
    user_imprs[uid].append([tsp, his, pos_imp, neg_imp, 0, uid])

for l in tqdm(open(data_path / "valid" / "behaviors.tsv", "r")):
    imp_id, uid, t, his, imprs = l.strip("\n").split("\t")
    his = his.split()
    tsp = t
    # tsp = time.mktime(time.strptime(t, "%m/%d/%Y %I:%M:%S %p"))
    #tsp = int(t)
    imprs = [i.split("-") for i in imprs.split(" ")]
    neg_imp = [i[0] for i in imprs if i[1] == "0"]
    pos_imp = [i[0] for i in imprs if i[1] == "1"]
    user_imprs[uid].append([tsp, his, pos_imp, neg_imp, 1, uid])

if os.path.exists(data_path / "test"):
    for l in tqdm(open(data_path / "test" / "behaviors.tsv", "r")):
        imp_id, uid, t, his, imprs = l.strip("\n").split("\t")
        his = his.split()
        #tsp = int(t)
        tsp = t
        # tsp = time.mktime(time.strptime(t, "%m/%d/%Y %I:%M:%S %p"))
        imprs = [i.split("-") for i in imprs.split(" ")]
        neg_imp = [i[0] for i in imprs if i[1] == "0"]
        pos_imp = [i[0] for i in imprs if i[1] == "1"]
        user_imprs[uid].append([tsp, his, pos_imp, neg_imp, 2, uid])


train_samples = []
valid_samples = []
test_samples = []
user_indices = defaultdict(list)

index = 0
for uid in tqdm(user_imprs):
    for impr in user_imprs[uid]:
        tsp, his, poss, negs, is_valid, uid = impr
        his = his[-max_his_len:]
        if is_valid == 0:
            for pos in poss:
                train_samples.append([pos, negs, his, uid])
                user_indices[uid].append(index)
                index += 1
        elif is_valid == 1:
            valid_samples.append([poss, negs, his, uid])
        else:
            test_samples.append([poss, negs, his, uid])

print(len(train_samples), len(valid_samples), len(test_samples))

# save user files
with open(out_path / "train_sam_uid.pkl", "wb") as f:
    pickle.dump(train_samples, f)

with open(out_path / "valid_sam_uid.pkl", "wb") as f:
    pickle.dump(valid_samples, f)

with open(out_path / "test_sam_uid.pkl", "wb") as f:
    pickle.dump(test_samples, f)

with open(out_path / "user_indices.pkl", "wb") as f:
    pickle.dump(user_indices, f)

# stat of train user samples
train_user_samples = 0

for uid in tqdm(user_indices):
    train_user_samples += len(user_indices[uid])

print(train_user_samples / len(user_indices))
