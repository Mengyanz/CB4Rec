"""Run experiment. """

import math, os 
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_Sim, NRMS_IPS_Sim 
from algorithms.neural_greedy import SingleStageNeuralGreedy
from algorithms.neural_ucb import SingleStageNeuralUCB, TwoStageNeuralUCB, DummyTwoStageNeuralUCB, TwoStageNeuralUCB_zhenyu, SingleNerual_TwoStageUCB
from algorithms.hcb import HCB
from algorithms.phcb import pHCB
from algorithms.neural_linear import NeuralLinearTS, NeuralLinearUCB_UserDisjoint, NeuralGLMUCB_UserItemHybrid, NeuralBilinUCB_Hybrid
from algorithms.linucb import SingleStageLinUCB, GLMUCB
from algorithms.uniform_random import UniformRandom
from core.contextual_bandit import run_contextual_bandit
import pretty_errors
import pickle

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
    # from configs.mezhang_params import parse_args
    from configs.zhenyu_params import parse_args
    args = parse_args()
    print(args)

    # args.sim_path = 'pretrained_models/sim_nrms_bce_r14_ep6_thres038414'
    # args.sim_path = 'model/large/large.pkl'
    # args.sim_path = 'pretrained_models/sim_nrms_bce_r14_ep6_thres038414_copy'
    args.sim_path = '/home/thanhnt/projects/CB4Rec/pretrained_models/sim_emp_ips_nrms_normalized_r14_ep5'
    args.reward_type = 'soft' # Use comparison instead of Bernoulli
    # args.sim_threshold = 0.38414
    rec_batch_size = 10
    # construct a simulator
    simulator = NRMS_IPS_Sim(device, args, pretrained_mode=True)

    print('Debug args.algo:', args.algo)
    if args.algo_prefix == 'algo':
        args.algo_prefix = args.algo + '-topicUpdate' + str(args.topic_update_period) + '-ninfernece' + str(args.n_inference) + '-dynamic' + str(args.dynamic_aggregate_topic)
        # + '-' + str(args.n_trials) + '-' + str(args.T) 
    print('Debug args.algo_prefix: ', args.algo_prefix)
    print('Debug dynamic_aggregate_topic: ', args.dynamic_aggregate_topic)

    # construct a list of CB learners 
    if args.algo == 'single_neuralucb':
        learner = SingleStageNeuralUCB(device, args)
    elif args.algo == 'ts_neuralucb':
        args.topic_update_period = 1 # update topic each iteration
        learner = TwoStageNeuralUCB(device, args)
    # dummylearner = DummyTwoStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=n_inference)
    elif args.algo == 'singleneural_twostageucb':
        learner = SingleNerual_TwoStageUCB(device, args)
    elif args.algo == 'neural_linearts':
        learner = NeuralLinearTS(device, args)
    elif args.algo == 'neural_linearucb':
        learner = NeuralLinearUCB(device, args)
    elif args.algo == 'neuralglmucb_uihybrid':
        learner = NeuralGLMUCB_UserItemHybrid(device, args)
    elif args.algo == 'neuralbilinucb_hybrid':
        learner = NeuralBilinUCB_Hybrid(device, args)
    elif args.algo == 'greedy':
        learner = SingleStageNeuralGreedy(device, args)
    elif args.algo == 'single_linucb':
        args.update_period = 1 # update parameters each iteration
        learner = SingleStageLinUCB(device, args)
    elif args.algo == 'glmucb':
        args.update_period = 1 # update parameters each iteration
        # Debug
        # args.per_rec_score_budget = 1e8
        # args.num_selected_users = 1
        # args.T = 100
        # args.n_trials = 1
        # args.gamma = 0
        learner = GLMUCB(device, args)
    elif args.algo == 'neuralucb_neuralucb':
        print(args.topic_update_period)
        learner = TwoStageNeuralUCB_zhenyu(device, args)
    elif args.algo == 'uniform_random':
        args.algo_prefix = args.algo
        learner = UniformRandom(device,args)
    elif args.algo == 'hcb':
        args.update_period = 1
        root = pickle.load(open(os.path.join(args.root_data_dir, args.dataset, 'utils', 'my_tree.pkl'), 'rb'))
        learner = HCB(device, args, root)
    elif args.algo == 'phcb':
        args.update_period = 1
        root = pickle.load(open(os.path.join(args.root_data_dir, args.dataset, 'utils', 'my_tree.pkl'), 'rb'))
        learner = pHCB(device, args, root)
    else:
        raise NotImplementedError

    algos = [learner]
    # algos = [greedylearner, single_neuralucb_learner, ts_neuralucb_learner]

    # construct dataset
    # contexts = simulator.valid_samples 

    # runner 
    # h_actions, h_rewards = run_contextual_bandit(args, simulator, algos)
    run_contextual_bandit(args, simulator, algos)


if __name__ == '__main__':
    main()