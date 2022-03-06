"""Define all unit tests here. """

import math, os 
import numpy as np 
import torch 
import pickle 
from algorithms.nrms_sim import NRMS_Sim, NRMS_IPS_Sim
from algorithms.neural_ucb import DummyTwoStageNeuralUCB


from configs.thanh_params import parse_args
args = parse_args()

def test_NRMS_Sim(device):
    """Test NRMS_Sim"""
    # print(args)
    nrms = NRMS_Sim(device, args)

    # print(nrms.model)

    clicked_history_fn = '/home/thanhnt/data/MIND/large/utils/train_clicked_history.pkl'
    with open(clicked_history_fn, 'rb') as fo: 
        clicked_history = pickle.load(fo)

    nrms.set_clicked_history(clicked_history)

    # uids = ['U403465','U493092','U172654','U248125','U495159','U288476','U92329'] 
    uid = 'U403465'
    news_indexes = [0,1,2,3,4] 
    rewards = nrms.reward(uid, news_indexes)

    print(rewards)

def test_NRMS_Sim_train(device):
    nrms = NRMS_Sim(device, args, pretrained_mode=False)
    nrms.train()


def test_NRMS_IPS_Sim_train(device):
    nrms = NRMS_IPS_Sim(device, args, pretrained_mode=False)
    nrms.train()


def test_DummyTwoStageNeuralUCB(device): 
    cbln = DummyTwoStageNeuralUCB(device, args, rec_batch_size = 3)
    uid = 'U403465'
    news_indexes = [0,1,2,3,4] 
    # res = cbln.item_rec(uid, news_indexes)
    # print(res.shape)
    # print(res)
    tp, it = cbln.sample_actions(uid) 
    print(tp)
    print(it)

def test_PropensityScore(device): 
    from algorithms.propensity_score import PropensityScore 
    prop = PropensityScore(args, device)
    prop.train()


if __name__ == '__main__': 
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    test_NRMS_Sim(device)
    # test_DummyTwoStageNeuralUCB()
    # test_NRMS_Sim_train(device)

    # test_PropensityScore(device)

    # test_NRMS_IPS_Sim_train(device)