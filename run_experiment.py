"""Run experiment. """

import math, os 
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_Sim 
from algorithms.neural_greedy import SingleStageNeuralGreedy
from algorithms.neural_ucb import SingleStageNeuralUCB, TwoStageNeuralUCB, DummyTwoStageNeuralUCB, TwoStageNeuralUCB_zhenyu
from algorithms.linucb import SingleStageLinUCB
from core.contextual_bandit import run_contextual_bandit
import logging
# import pretty_errors

os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
device = torch.device("cuda:2")
torch.cuda.set_device(device)

def main():
    # from configs.thanh_params import parse_args
#     from configs.mezhang_params import parse_args
    from configs.zhenyu_params import parse_args
    args = parse_args()

    log_path = os.path.join(args.root_proj_dir, 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path) 
    logging.basicConfig(filename=os.path.join(log_path, '{}.log'.format(args.algo)), level=logging.INFO)
    logging.info(args)

    rec_batch_size = 5
    per_rec_score_budget = 1000
    n_inference = 5
    args.sim_path = 'pretrained_models/sim_nrms_bce_r14_ep6_thres038414'
    args.sim_threshold = 0.38414

    # construct a simulator
    simulator = NRMS_Sim(device, args)

    print('Debug args.algo:', args.algo)

    # construct a list of CB learners 
    if args.algo == 'single_neuralucb':
        learner = SingleStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=n_inference, per_rec_score_budget = per_rec_score_budget)
    elif args.algo == 'ts_neuralucb':
        learner = TwoStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=n_inference, per_rec_score_budget = per_rec_score_budget)
    # dummylearner = DummyTwoStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=n_inference)
    elif args.algo == 'greedy':
        learner = SingleStageNeuralGreedy(device, args, rec_batch_size = rec_batch_size, per_rec_score_budget = per_rec_score_budget)
    elif args.algo == 'single_linucb':
        learner = SingleStageLinUCB(device, args, rec_batch_size = rec_batch_size, per_rec_score_budget = per_rec_score_budget)
    elif args.algo == 'ts_neuralucb_zhenyu':
        learner = TwoStageNeuralUCB_zhenyu(device, args, rec_batch_size = rec_batch_size)
    else:
        raise NotImplementedError

    algos = [learner]
    # algos = [greedylearner, single_neuralucb_learner, ts_neuralucb_learner]
    for learner in algos:
        logging.info(learner.name)

    # construct dataset
    # contexts = simulator.valid_samples 

    # runner 
    h_actions, h_rewards = run_contextual_bandit(args, simulator, rec_batch_size, algos)


if __name__ == '__main__':
    main()