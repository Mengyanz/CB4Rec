"""Run experiment. """

import math, os 
from absl import flags, app
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_IPS_Sim 
from algorithms.neural_ucb import SingleStageNeuralUCB, TwoStageNeuralUCB, DummyTwoStageNeuralUCB
from core.contextual_bandit import run_contextual_bandit
from configs.thanh_params import parse_args

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda:0")
torch.cuda.set_device(device)

parser = parse_args() 
args = parser.parse_args()

args.T = 100 
args.n_trials = 5
# args.sim_path = 'pretrained_models/sim_nrms_bce_r14_ep6_thres038414'
# args.sim_path = 'model/large/large.pkl'
args.sim_path = '/home/thanhnt/projects/CB4Rec/pretrained_models/sim_emp_ips_nrms_normalized_r14_ep5'
args.sim_threshold = 0.38414

print(args)

def main():
    # args.reward_type = 'hard' # hard/soft/hybrid/threshold/threshold_eps/bern
    # args.topic_update_disabled = False
    rec_batch_size = 10

    simulator = NRMS_IPS_Sim(device, args, pretrained_mode=True)

    dummylearner = DummyTwoStageNeuralUCB(device, args, rec_batch_size = rec_batch_size, n_inference=3,pretrained_mode = args.pretrained_cb )

    algos = [dummylearner]

    h_actions, h_rewards = run_contextual_bandit(args, simulator, rec_batch_size, algos)

if __name__ == '__main__':
    main()
