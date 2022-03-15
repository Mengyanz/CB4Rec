"""Run experiment. """

import math, os 
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_Sim 
from algorithms.neural_ucb import SingleStageNeuralUCB, TwoStageNeuralUCB, DummyTwoStageNeuralUCB
from core.contextual_bandit import run_contextual_bandit

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda:2")
torch.cuda.set_device(device)

def main():
    from configs.thanh_params import parse_args
    # from configs.mezhang_params import parse_args

    args = parse_args()

    # args.sim_path = 'pretrained_models/sim_nrms_bce_r14_ep6_thres038414'
    # args.sim_path = 'model/large/large.pkl'
    # args.sim_path = 'pretrained_models/sim_nrms_bce_r14_ep6_thres038414_copy'
    args.sim_path = '/home/thanhnt/projects/CB4Rec/pretrained_models/sim_emp_ips_nrms_normalized_r14_ep5'
    args.reward_type = 'soft' # Use comparison instead of Bernoulli
    # args.sim_threshold = 0.38414
    rec_batch_size = 10
    # construct a simulator
    simulator = NRMS_Sim(device, args)

    # construct a list of CB learners 
    # ucblearner = SingleStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=10)
    # ucblearner = TwoStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=3)
    dummylearner = DummyTwoStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=3)

    algos = [dummylearner]

    # construct dataset
    # contexts = simulator.valid_samples 

    # runner 
    h_actions, h_rewards = run_contextual_bandit(args, simulator, rec_batch_size, algos)

if __name__ == '__main__':
    main()