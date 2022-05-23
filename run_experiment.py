"""Run experiment. """

import math, os 
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_Sim, NRMS_IPS_Sim 
from algorithms.neural_greedy import NeuralGreedy, Two_NeuralGreedy
from algorithms.neural_ucb import NeuralDropoutUCB, Two_NeuralDropoutUCB
# from algorithms.hcb import HCB
# from algorithms.phcb import pHCB
from algorithms.neural_linear import NeuralLinUCB, NeuralGLMUCB, NeuralGLMAddUCB
from algorithms.neural_bilinear import NeuralGBiLinUCB
from algorithms.proposed import Two_NeuralGLMAddUCB, Two_NeuralGBiLinUCB
from algorithms.linucb import LinUCB, GLMUCB
from algorithms.uniform_random import UniformRandom, Two_Random
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
    from configs.params import parse_args
    args = parse_args()
    args.root_data_dir = os.path.join(args.root_dir, args.root_data_dir)
    args.root_proj_dir = os.path.join(args.root_dir, args.root_proj_dir)
    args.result_path = os.path.join(args.root_dir, args.result_path)
    
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
    elif args.algo == 'neural_glmadducb':
        learner = NeuralGLMAddUCB(args, device)
    elif args.algo == 'neural_gbilinucb':
        args.glm_lr = 1e-3
        learner = NeuralGBiLinUCB(args, device)
    # elif args.algo == 'hcb':
    #     args.update_period = 1
    #     root = pickle.load(open(os.path.join(args.root_data_dir, args.dataset, 'utils', 'my_tree.pkl'), 'rb'))
    #     learner = HCB(device, args, root)
    # elif args.algo == 'phcb':
    #     args.update_period = 1
    #     root = pickle.load(open(os.path.join(args.root_data_dir, args.dataset, 'utils', 'my_tree.pkl'), 'rb'))
    #     learner = pHCB(device, args, root)
    # ----------------------------- Two stage ----------------------------------#
    elif args.algo == '2_random':
        learner = Two_Random(args, device)
    elif args.algo == '2_neuralgreedy':
        learner = Two_NeuralGreedy(args, device)
    elif args.algo == '2_ts_neuralucb':
        args.topic_update_period = 1 # update topic each iteration
        learner = ThompsonSampling_NeuralDropoutUCB(args, device)
    elif args.algo == '2_neuralucb':
        learner = Two_NeuralDropoutUCB(args, device)
    elif args.algo == '2_neuralglmadducb':
        learner = Two_NeuralGLMAddUCB(args, device)
    elif args.algo == '2_neuralglmbilinucb':
        args.glm_lr = 1e-3
        learner = Two_NeuralGBiLinUCB(args, device)
    else:
        raise NotImplementedError

    algos = [learner]
    print(args)
    args_save_path = os.path.join(args.result_path, args.algo_prefix)
    with open(args_save_path, 'wt') as f:
        json.dump(vars(args), f, indent=4)
    run_contextual_bandit(args, simulator, algos)


if __name__ == '__main__':
    main()