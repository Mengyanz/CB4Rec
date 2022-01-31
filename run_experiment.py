"""Run experiment. """

import math, os 
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_Sim 
from algorithms.neural_greedy import SingleStageNeuralGreedy
from algorithms.neural_ucb import SingleStageNeuralUCB, TwoStageNeuralUCB, DummyTwoStageNeuralUCB
from core.contextual_bandit import run_contextual_bandit
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda:2")
torch.cuda.set_device(device)

def main():
    # from configs.thanh_params import parse_args
    from configs.mezhang_params import parse_args
    args = parse_args()

    log_path = os.path.join(args.root_proj_dir, 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path) 
    logging.basicConfig(filename=os.path.join(log_path, 'mylog.log'), level=logging.INFO)
    logging.info(args)

    rec_batch_size = 3
    n_inference = 3
    # construct a simulator
    simulator = NRMS_Sim(device, args)

    # construct a list of CB learners 
    # ucblearner = SingleStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=10)
    # ucblearner = TwoStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=3)
    dummylearner = DummyTwoStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=n_inference)
    greedylearner = SingleStageNeuralGreedy(device, args, rec_batch_size = rec_batch_size)

    algos = [dummylearner]
    for learner in algos:
        logging.info(learner.name)

    # construct dataset
    # contexts = simulator.valid_samples 

    # runner 
    h_actions, h_rewards = run_contextual_bandit(args, simulator, rec_batch_size, algos)


if __name__ == '__main__':
    
    main()