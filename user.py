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
from datetime import datetime 
date_format_str = '%m/%d/%Y %I:%M:%S %p'

# config
data_path = Path("/home/v-mezhang/blob/data/demo")
out_path = Path("/home/v-mezhang/blob/data/demo/utils")

train_imprs_path = os.path.join(data_path,"train","behaviors.tsv")
valid_imprs_path = os.path.join(data_path,"valid","behaviors.tsv")
test_imprs_path = os.path.join(data_path,"test","behaviors.tsv")
print(train_imprs_path)

npratio = 4
max_his_len = 50
min_word_cnt = 3
max_title_len = 30
user_imprs = defaultdict(list)

def read_imprs(file_path, user_imprs, max_his_len, mode):
    index = 0
    samples = []
    user_indices = defaultdict(list)

    for l in tqdm(open(file_path, "r")):
        imp_id, uid, t, his, imprs = l.strip("\n").split("\t")
        his = his.split()
        tsp = t
        # tsp = time.mktime(time.strptime(t, "%m/%d/%Y %I:%M:%S %p"))
        #tsp = int(t)
        imprs = [i.split("-") for i in imprs.split(" ")]
        neg_imp = [i[0] for i in imprs if i[1] == "0"]
        pos_imp = [i[0] for i in imprs if i[1] == "1"]
        user_imprs[uid].append([tsp, his, pos_imp, neg_imp, mode, uid])

        his = his[-max_his_len:]
        if mode == 0:
            for pos in pos_imp:
                samples.append([pos, neg_imp, his, uid, tsp])
                user_indices[uid].append(index)
                index += 1
        else:
            samples.append([pos_imp, neg_imp, his, uid, tsp])


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

    return samples, user_indices 


train_samples, train_user_indices = read_imprs(train_imprs_path, user_imprs, max_his_len, 0)

valid_samples, valid_user_indices = read_imprs(valid_imprs_path, user_imprs, max_his_len, 1)
sorted_valid_samples = [i for i in sorted(valid_samples, key=lambda date: datetime.strptime(date[-1], date_format_str))]
with open(out_path / "sorted_valid_sam_uid.pkl", "wb") as f:
    pickle.dump(sorted_valid_samples, f)

print(len(train_samples), len(valid_samples))

if os.path.exists(test_imprs_path):
    test_samples, test_user_indices = read_imprs(test_imprs_path, user_imprs, max_his_len, 2)
    sorted_test_samples = [i for i in sorted(test_samples, key=lambda date: datetime.strptime(date[-1], date_format_str))]
    with open(out_path / "sorted_test_sam_uid.pkl", "wb") as f:
        pickle.dump(sorted_test_samples, f)
    print(len(test_samples))

# i = 0
# for s in sorted_valid_samples:
#     print(s[-1])
#     i += 1

#     if i > 10: 
#         break

# raise Exception


# index = 0
# for uid in tqdm(user_imprs):
#     for impr in user_imprs[uid]:
#         tsp, his, poss, negs, is_valid, uid = impr
#         his = his[-max_his_len:]
#         if is_valid == 0:
#             for pos in poss:
#                 train_samples.append([pos, negs, his, uid])
#                 user_indices[uid].append(index)
#                 index += 1
#         elif is_valid == 1:
#             valid_samples.append([poss, negs, his, uid])
#         else:
#             test_samples.append([poss, negs, his, uid])


# stat of train user samples
train_user_samples = 0

for uid in tqdm(train_user_indices):
    train_user_samples += len(train_user_indices[uid])

print(train_user_samples / len(train_user_indices))
