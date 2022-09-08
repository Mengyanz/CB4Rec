"""Run experiment. """
import math, os 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_IPS_Sim, NRMS_Sim
from algorithms.propensity_score import PropensityScore

# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4"



def main():
    from configs.t_params import parse_args
    args = parse_args()
    args.root_data_dir = os.path.join(args.root_dir, args.root_data_dir)
    args.root_proj_dir = os.path.join(args.root_dir, args.root_proj_dir)

    args.ips_normalize = True 
    args.empirical_ips = True # True if use empirical IPS, False if use PropensityScoreModel 
    # print(args)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if not args.empirical_ips: # train a propensity model first if it is not available
        PropensityScore(args, device)

    nrms = NRMS_IPS_Sim(device, args, pretrained_mode=False, train_mode=True)
    # nrms = NRMS_Sim(device, args, pretrained_mode=False)
    nrms.train()



if __name__ == '__main__':
    main()