"""Run experiment. """

import math, os 
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_Sim 
from algorithms.neural_ucb import SingleStageNeuralUCB, TwoStageNeuralUCB
from core.contextual_bandit import run_contextual_bandit

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda:0")
torch.cuda.set_device(device)

def main():
    # from configs.thanh_params import parse_args
    from configs.mezhang_params import parse_args

    args = parse_args()

    rec_batch_size = 2
    # construct a simulator
    simulator = NRMS_Sim(device, args)

    # construct a list of CB learners 
    # ucblearner = SingleStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=10)
    ucblearner = TwoStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=3)
    algos = [ucblearner]

    # construct dataset
    contexts = simulator.valid_samples 

    # runner 
    h_actions, h_rewards = run_contextual_bandit(args, contexts, simulator, rec_batch_size, algos)


if __name__ == '__main__':
    main()