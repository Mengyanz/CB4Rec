"""Run experiment. """

import math, os 
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_IPS_Sim


# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4"



def main():
    from CB4Rec.configs.t_params import parse_args
    args = parse_args()
    args.ips_normalize = True 
    args.empirical_ips = False #True
    print(args)

    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    nrms = NRMS_IPS_Sim(device, args, pretrained_mode=False, train_mode=True)
    nrms.train()



if __name__ == '__main__':
    main()