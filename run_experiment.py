"""Run experiment. """

import math, os 
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_Sim, NRMS_IPS_Sim 
from algorithms.neural_greedy import NeuralGreedy
from algorithms.neural_ucb import NeuralDropoutUCB, ThompsonSampling_NeuralDropoutUCB, DummyThompsonSampling_NeuralDropoutUCB, NeuralDropoutUCB_NeuralDropoutUCB
from algorithms.hcb import HCB
from algorithms.phcb import pHCB
from algorithms.neural_linear import NeuralLinUCB, NeuralGLMUCB, NeuralGLMUCB_Newton, NeuralGLMUCB_LBFGS, NeuralGLMAddUCB
from algorithms.neural_bilinear import NeuralGBiLinUCB
from algorithms.linucb import LinUCB, GLMUCB
from algorithms.uniform_random import UniformRandom
from core.contextual_bandit import run_contextual_bandit
import pretty_errors
import pickle
import json

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda:0")
torch.cuda.set_device(device)

class Tree():
    def __init__(self):
        self.emb         = None
        self.size        = 0
        self.gids        = []
        self.children    = None
        self.is_leaf     = False

def main():
    # from configs.thanh_params import parse_args
    from configs.mezhang_params import parse_args
    # from configs.zhenyu_params import parse_args
    args = parse_args()
    
    # args.sim_path = 'pretrained_models/sim_nrms_bce_r14_ep6_thres038414'
    # args.sim_path = 'model/large/large.pkl'
    # args.sim_path = 'pretrained_models/sim_nrms_bce_r14_ep6_thres038414_copy'
    # args.sim_path = './pretrained_models/sim_emp_ips_nrms_normalized_r14_ep5'
    # args.sim_threshold = 0.38414
    
    # construct a simulator
    simulator = NRMS_IPS_Sim(device, args, pretrained_mode=True)

    print('Debug args.algo:', args.algo)
    if args.algo_prefix == 'algo':
        args.algo_prefix = args.algo + '-topicUpdate' + str(args.topic_update_period) + '-ninfernece' + str(args.n_inference) + '-dynamic' + str(args.dynamic_aggregate_topic)
        # + '-' + str(args.n_trials) + '-' + str(args.T) 
    print('Debug args.algo_prefix: ', args.algo_prefix)

     # rec_batch_size = 10
    # dummylearner = DummyThompsonSampling_NeuralDropoutUCB(device, args, rec_batch_size = rec_batch_size, n_inference=n_inference)

    # construct a list of CB learners 
    # ----------------------------- Two stage ----------------------------------#
    if args.algo == 'uniform_random':
        learner = UniformRandom(args, device)
    elif args.algo == 'greedy':
        learner = NeuralGreedy(args, device)
    elif args.algo == 'neural_dropoutucb':
        learner = NeuralDropoutUCB(args, device) 
    elif args.algo == 'linucb':
        args.update_period = 1 # update parameters each iteration
        learner = LinUCB(args, device)
    elif args.algo == 'glmucb':
        args.update_period = 1 # update parameters each iteration
        learner = GLMUCB(args, device)
    elif args.algo == 'neural_linucb':
        learner = NeuralLinUCB(args, device)
    elif args.algo == 'neural_glmucb':
        learner = NeuralGLMUCB(args, device)
    elif args.algo == 'neural_glmucb_newton':
        learner = NeuralGLMUCB_Newton(args, device)
    elif args.algo == 'neural_glmucb_lbfgs':
        learner = NeuralGLMUCB_LBFGS(args, device)
    elif args.algo == 'neural_glmadducb':
        learner = NeuralGLMAddUCB(args, device)
    elif args.algo == 'neural_gbilinucb':
        learner = NeuralGBiLinUCB(args, device)
    elif args.algo == 'hcb':
        args.update_period = 1
        root = pickle.load(open(os.path.join(args.root_data_dir, args.dataset, 'utils', 'my_tree.pkl'), 'rb'))
        learner = HCB(device, args, root)
    elif args.algo == 'phcb':
        args.update_period = 1
        root = pickle.load(open(os.path.join(args.root_data_dir, args.dataset, 'utils', 'my_tree.pkl'), 'rb'))
        learner = pHCB(device, args, root)
    # ----------------------------- Two stage ----------------------------------#
    elif args.algo == '2_ts_neuralucb':
        args.topic_update_period = 1 # update topic each iteration
        learner = ThompsonSampling_NeuralDropoutUCB(args, device)
    elif args.algo == '2_neuralucb_neuralucb':
        print(args.topic_update_period)
        learner = NeuralDropoutUCB_NeuralDropoutUCB(args, device)
    else:
        raise NotImplementedError

    algos = [learner]
    # algos = [greedylearner, neural_dropoutucb_learner, 2_ts_neuralucb_learner]

    # construct dataset
    # contexts = simulator.valid_samples 

    # runner 
    # h_actions, h_rewards = run_contextual_bandit(args, simulator, algos)
    print(args)
    args_save_path = os.path.join(args.result_path, args.algo_prefix)
    with open(args_save_path, 'wt') as f:
        json.dump(vars(args), f, indent=4)
    run_contextual_bandit(args, simulator, algos)


if __name__ == '__main__':
    main()