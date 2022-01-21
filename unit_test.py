"""Define all unit tests here. """

import math, os 
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_Sim 

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda:0")
torch.cuda.set_device(device)

print(device)

def test_NRMS_Sim():
    """Test NRMS_Sim"""
    from configs.thanh_params import parse_args
    args = parse_args()
    print(args)
    nrms = NRMS_Sim(device, args)

    print(nrms.model)
    print(nrms.news_index.shape)

    news_vec = nrms.get_news_vec(1)
    print(news_vec)
    print(news_vec.shape)


if __name__ == '__main__': 
    test_NRMS_Sim()