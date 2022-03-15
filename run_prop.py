
import math, os 
import numpy as np 
import torch 
import pickle 

from algorithms.propensity_score import PropensityScore 
from configs.thanh_params import parse_args
args = parse_args()
args.ips_path = "/home/thanhnt/projects/CB4Rec/runs/propmodel_pn=2-10/model_1"
device = torch.device("cuda:0")
torch.cuda.set_device(device)

prop = PropensityScore(args, device, pretrained_mode=True)
prop.train_with_resume(epoch_number=2, eval=False)