"""Run experiment. """
import math, os 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np 
import torch 
from algorithms.nrms_sim import NRMS_IPS_Sim, NRMS_Sim
from algorithms.propensity_score import PropensityScore
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4"

def plot_threshold(args):
    y_scores = np.load(os.path.join(args.sim_path, 'preds.npy'))
    y_trues = np.load(os.path.join(args.sim_path, 'trues.npy'))
    imp_metrics = np.load(os.path.join(args.sim_path, 'perimp_metrics.npy'))
    plt.hist(y_scores[y_trues == 1], alpha = 0.5, color = 'r', density = True, bins=100, label = '1')
    plt.hist(y_scores[y_trues == 0], alpha = 0.5, color = 'b', density = True, bins=100, label = '0')
    plt.legend()
    plt.savefig(os.path.join(args.sim_path, 'threshold.png'))

    precision, recall, thresholds = precision_recall_curve(y_trues, y_scores)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    print(' Best Threshold=%f, fscore=%.3f' % (thresholds[ix], fscore[ix]))
    print('AUC: {}'.format(np.mean(imp_metrics, axis=0)[0]))



def main():
    from configs.params import parse_args
    args = parse_args()
    args.root_data_dir = os.path.join(args.root_dir, args.root_data_dir)
    args.root_proj_dir = os.path.join(args.root_dir, args.root_proj_dir)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    args.ips_normalize = True 
    args.empirical_ips = True # True if use empirical IPS, False if use PropensityScoreModel 
    # print(args)

    if args.dataset == 'large':
        args.lr = 1e-4
    # elif args.dataset == 'adressa':
    #     args.lr =1e-5 # 0.000001   
    elif args.dataset == 'movielens':
        args.lr = 1e-4

    # args.sim_path = './pretrained_models/sim_nrms_adressa_r14_ep5'
    # nrms = NRMS_Sim(device, args, pretrained_mode=True)
    # args.sim_path = './pretrained_models/sim_emp_ips_nrms_normalized_adressa_r14_ep6'
    # nrms = NRMS_IPS_Sim(device, args, pretrained_mode=True, train_mode=False)

    # args.sim_path = './pretrained_models/sim_nrms_mv_r14_ep74'
    # nrms = NRMS_Sim(device, args, pretrained_mode=True)
    # args.sim_path = './pretrained_models/sim_emp_ips_nrms_normalized_mv_r14_ep86'
    # nrms = NRMS_IPS_Sim(device, args, pretrained_mode=True, train_mode=False)

    if not args.empirical_ips: # train a propensity model first if it is not available
        PropensityScore(args, device)

    nrms = NRMS_IPS_Sim(device, args, pretrained_mode=False, train_mode=True)
    # nrms = NRMS_Sim(device, args, pretrained_mode=False)
    nrms.train()
    nrms.evaluate()
    # plot_threshold(args)


if __name__ == '__main__':
    main()