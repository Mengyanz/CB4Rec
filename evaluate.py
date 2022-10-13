"""Run evaluation. """

import math, os, sys
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from collections import defaultdict
import glob 
from utils.data_util import load_word2vec
import torch 
from torch.utils.data import DataLoader
from algorithms.nrms_sim import NRMS_IPS_Sim 
from utils.data_util import NewsDataset, TrainDataset, load_word2vec
from utils.metrics import ILAD, ILMD

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

rcParams['axes.labelsize'] = MEDIUM_SIZE # 15
rcParams['xtick.labelsize'] = SMALL_SIZE # 13
rcParams['ytick.labelsize'] = SMALL_SIZE # 13
rcParams['legend.fontsize'] = SMALL_SIZE # 13
rcParams['axes.titlesize'] = BIGGER_SIZE  # 15

def get_sim_news_embs(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    _, _, nindex2vec = load_word2vec(args, 'utils')
    news_dataset = NewsDataset(nindex2vec) 
    news_dl = DataLoader(news_dataset,batch_size=1024, shuffle=False, num_workers=args.num_workers)

    simulator = NRMS_IPS_Sim(device, args, pretrained_mode=True) 

    simulator.model.eval()
    with torch.no_grad():
        news_vecs = []
        for news in news_dl: 
            news = news.to(device)
            news_vec = simulator.model.text_encoder(news).detach().cpu().numpy()
            news_vecs.append(news_vec)
    return np.concatenate(news_vecs)

def cal_diversity(args, rec_items_all, algo_names, metric_names = ['ILAD-batch', 'ILAD-trial']):
    n_trials, n_algos, rec_bs, T = rec_items_all.shape
    print('# trails: ', n_trials, ', # algos: ', n_algos, ', # rec_bs: ', rec_bs, ', T: ', T)
    rec_items_all = rec_items_all.transpose(1, 0, 3, 2) # n_algos, n_trials, T, rec_bs
    news_vecs = get_sim_news_embs(args)
    
    metrics = {}
    batch_means = []
    batch_stds = []
    trial_means = []
    trial_stds = []

    for algo_i in range(rec_items_all.shape[0]):
        all_batch_ILADs = [] # diversity for batch
        all_trial_ILADs = [] # diversity for trial
        for trial_i in range(rec_items_all.shape[1]):
            batch_ILADs = [] # diversity for batch 
            trial_ILADs = []
            rec_items = rec_items_all[algo_i][trial_i]
            all_embedding = []
            for i, batch in enumerate(rec_items):
                batch = list(map(int, batch))
                nv = news_vecs[batch]
                nv = nv/np.sqrt(np.square(nv).sum(axis=-1)).reshape((nv.shape[0],1))
                all_embedding.append(nv)
                if (i + 1) % 100 == 0:
                    batch_ILADs.append(ILAD(nv))
                    trial_ILAD = ILAD(np.concatenate(all_embedding,axis=0))
                    trial_ILADs.append(trial_ILAD)

            all_batch_ILADs.append(batch_ILADs)
            all_trial_ILADs.append(trial_ILADs)
            
            # all_batch_ILADs.append(np.mean(batch_ILADs))
            # all_embedding = np.concatenate(all_embedding, axis=0)
            # all_trial_ILADs.append(ILAD(all_embedding))

        all_batch_ILADs = np.array(all_batch_ILADs) # n_trial, T
        all_trial_ILADs = np.array(all_trial_ILADs) # n_trial, T
            
        batch_means.append(np.mean(all_batch_ILADs, axis = 0))
        batch_stds.append(np.std(all_batch_ILADs, axis = 0))
        trial_means.append(np.mean(all_trial_ILADs, axis = 0))
        trial_stds.append(np.std(all_trial_ILADs, axis = 0))
  
        print('Algorithm: ', algo_names[algo_i])
        print('ILAD-batch Mean: {0:.3f}'.format(batch_means[algo_i].mean()))
        print('ILAD-batch Std: {0:.3f}'.format(batch_stds[algo_i].mean()))
        print('ILAD-trial Mean: {0:.3f}'.format(trial_means[algo_i][-1]))
        print('ILAD-trial Std: {0:.3f}'.format(trial_stds[algo_i][-1]))
        print()

    metrics['ILAD-batch'] = [np.array(batch_means), np.array(batch_stds)]
    metrics['ILAD-trial'] = [np.array(trial_means), np.array(trial_stds)]
    return metrics


def cal_metric(h_rewards_all, algo_names, metric_names = ['cumu_reward']):  
    n_trials, n_algos, rec_bs, T = h_rewards_all.shape
    print('# trails: ', n_trials, ', # algos: ', n_algos, ', # rec_bs: ', rec_bs, ', T: ', T)
    metrics = {}
    
    for metric in metric_names:
        print('-----------Metric: {}------------'.format(metric))
        if metric == 'cumu_reward':
            cumu_rewards = np.cumsum(np.sum(h_rewards_all, axis=2), axis = -1) # n_trails, n_algos, T
            cumu_rewards_mean = np.mean(cumu_rewards, axis = 0) # n_algos, T
            cumu_rewards_std = np.std(cumu_rewards, axis = 0) # n_algos, T

            for i in range(n_algos): 
                print('Algorithm: ', algo_names[i])
                print('Mean: {0:.3f}'.format(cumu_rewards_mean[i][-1]))
                print('Std: {0:.3f}'.format(cumu_rewards_std[i][-1]))
                print()

            metrics[metric] = [cumu_rewards_mean, cumu_rewards_std]

        if metric == 'cumu_ctr':
            cumu_rewards = np.cumsum(np.mean(h_rewards_all, axis=2), axis = -1) # n_trails, n_algos, T
            cumu_rewards_mean = np.mean(cumu_rewards, axis = 0) # n_algos, T
            cumu_rewards_std = np.std(cumu_rewards, axis = 0) # n_algos, T

            for i in range(n_algos): 
                print('Algorithm: ', algo_names[i])
                print('Mean: {0:.3f}'.format(cumu_rewards_mean[i][-1]))
                print('Std: {0:.3f}'.format(cumu_rewards_std[i][-1]))
                print()

            metrics[metric] = [cumu_rewards_mean, cumu_rewards_std]
            
        if metric == 'cumu_regret':
            cumu_rewards = np.cumsum(np.mean(h_rewards_all, axis=2), axis = -1) # n_trails, n_algos, T
            cumu_rewards_mean = np.mean(cumu_rewards, axis = 0) # n_algos, T
            cumu_rewards_std = np.std(cumu_rewards, axis = 0) # n_algos, T
            
            cumu_opt_rewards = np.asarray([[0.9 * i for i in range(1, T+1)] for _ in range(n_algos)])
            cumu_regret = np.subtract(cumu_opt_rewards, cumu_rewards_mean)
            
          
            metrics[metric] = [cumu_regret, cumu_rewards_std]
            
        if metric == 'moving_ctr':
            window_size = 100
            moving_averages = np.zeros((n_trials, n_algos, T-window_size+1))
            ctr = np.mean(h_rewards_all, axis=2) # n_trails, n_algos, T
            while i < T - window_size + 1:
                window = ctr[:,:,i:i+window_size]
                window_average = np.mean(window, axis = -1) 
      
                # Store the average of current
                # window in moving average list
                moving_averages[:,:,i] = window_average
                
                # Shift window to right by one position
                i += 1
                
            cumu_rewards_mean = np.mean(moving_averages, axis = 0) # n_algos, T-window_size
            cumu_rewards_std = np.std(moving_averages, axis = 0) # n_algos, T-window_size

            for i in range(n_algos): 
                print('Algorithm: ', algo_names[i])
                print('Mean: {0:.3f}'.format(cumu_rewards_mean[i][-1]))
                print('Std: {0:.3f}'.format(cumu_rewards_std[i][-1]))
                print()

            metrics[metric] = [cumu_rewards_mean, cumu_rewards_std]


        if metric == 'ctr':
            ave_ctr = np.cumsum(np.mean(h_rewards_all, axis=2), axis = -1)/np.arange(1, T+1) # n_trails, n_algos, T
            ave_ctr_mean = np.mean(ave_ctr, axis = 0) # n_algos, T
            ave_ctr_std = np.std(ave_ctr, axis = 0) # n_algos, T

            for i in range(n_algos):
                print('Algorithm: ', algo_names[i])
                print('Mean: {0:.3f}'.format(ave_ctr_mean[i][-1]))
                print('Std: {0:.3f}'.format(ave_ctr_std[i][-1]))
                print()

            metrics[metric] = [ave_ctr_mean, ave_ctr_std]

        if metric == 'ave_ctr':
            ave_ctr = np.mean(h_rewards_all, axis=2)
            ave_ctr_mean = np.mean(ave_ctr, axis = 0) # n_algos, T
            ave_ctr_std = np.std(ave_ctr, axis = 0) # n_algos, T

            for i in range(n_algos):
                print('Algorithm: ', algo_names[i])
                print('Mean: {0:.3f}'.format(ave_ctr_mean[i][-1]))
                print('Std: {0:.3f}'.format(ave_ctr_std[i][-1]))
                print()

            metrics[metric] = [ave_ctr_mean, ave_ctr_std]


        if metric == 'raw_reward':
            cumu_rewards = np.sum(h_rewards_all, axis=2)
            cumu_rewards_mean = np.mean(cumu_rewards, axis = 0) # n_algos, T
            cumu_rewards_std = np.std(cumu_rewards, axis = 0) # n_algos, T

            for i in range(n_algos):
                print('Algorithm: ', algo_names[i])
                print('Mean: {0:.3f}'.format(cumu_rewards_mean[i][-1]))
                print('Std: {0:.3f}'.format(cumu_rewards_std[i][-1]))
                print()

            metrics[metric] = [cumu_rewards_mean, cumu_rewards_std]
    
    return metrics

def plot_metrics(args, eva_path, metrics, algo_names, plot_title=None, save_title = None):
    # plt_path = os.path.join(args.root_proj_dir, 'plots')
    yaxis_dict = {'cumu_ctr': 'Cumulative CTR',
                'ctr': 'CTR',
                'moving_ctr': 'Moving CTR',
                'cumu_regret': 'Cumulative Regret'
    }
    label_dict = {'neural_glmadducb': 'S-N-GALM',
                'neural_gbilinucb': 'S-N-GBLM',
                '2_neuralglmadducb': '2-S-N-GALM',
                '2_neuralglmbilinucb': '2-S-N-GBLM'
    }
    title_dict = {
        'large': 'MIND',
        'movielens': 'MovieLens-20M'
    }
    
    for name, value in metrics.items():
        plt.figure()
        mean, std = value
        n_algos, T = mean.shape
        for i in range(n_algos):
            label_name = algo_names[i].split('_recSize')[0]
            if label_name in label_dict:
                label_name = label_dict[label_name]
            if name == 'raw_reward' or name == 'ave_ctr':
                # if i == 3:
                plt.scatter(range(T), mean[i], label = label_name, s = 1, alpha = 0.5)
            elif 'ILAD' in name:
                x = list(range(T * 100 +100)[::100])[1:]
                plt.plot(x, mean[i], label = label_name)
                plt.fill_between(x, mean[i] + std[i], mean[i] - std[i], alpha = 0.2)
            elif name == 'moving_ctr':
                plt.plot(range(T)[::10], mean[i][::10], label = label_name)
                plt.fill_between(range(T), mean[i] + std[i], mean[i] - std[i], alpha = 0.2)
            else:
                # plt.plot(range(T)[8000:], mean[i][8000:], label = label_name)
                # plt.fill_between(range(T)[8000:], (mean[i] + std[i])[8000:], (mean[i] - std[i])[8000:], alpha = 0.2)
                plt.plot(range(T), mean[i], label = label_name)
                plt.fill_between(range(T), (mean[i] + std[i]), (mean[i] - std[i]), alpha = 0.2)
        
        if name == 'ctr' or name == 'ave_ctr':
            plt.ylim(0,1)
            plt.plot([0, 1999], [0.075, 0.075], label = 'uniform_random', color = 'grey', linestyle='--')
        # elif name == 'cumu_ctr':
        #     # plt.ylim(-50,2050)
        #     plt.ylim(-50,10000)
        # plt.legend(bbox_to_anchor=(1.04,0.5), loc='center left')
        plt.legend()
        plt.xlabel('Iterations')
        if name in yaxis_dict:
            name = yaxis_dict[name]
        plt.ylabel(name)
        plt.title(title_dict[args.dataset])
        # plt.title(plot_title.replace('_', ' '))
        # plt.savefig(os.path.join(plt_path, save_title + '_' + name + '.pdf'),bbox_inches='tight')
        plt.savefig(os.path.join(eva_path, name + '.png'),bbox_inches='tight')


def cal_base_ctr(args):
    """calculate base ctr: uniform random policy 10 trails average ctr
    """
    filename = 'rewards-uniform_random-9-1000.npy'
    h_rewards_all = np.array(np.load(os.path.join(args.root_proj_dir, "results", filename)))
    print(h_rewards_all.shape)
    all_rewards = np.concatenate([h_rewards_all], axis = 1)
    metrics = cal_metric(all_rewards, 'uniform_random', ['cumu_reward', 'ctr'])
    plot_metrics(args, metrics, 'uniform_random', plot_title='Uniform Random 10 trials')
    return metrics

def collect_rewards(args, algo_group, timestr, algo_prefixes, algo_names, all_rewards, all_items, trials = '[0-9]', T =2000, n_trial = 5, rec_batch_size=5):
    """collect rewards 
    the rewards files are saved in the results/algo_group/timestr/trial-algo_prefix-round.npy
    each file stores array with shape (n_algos, rec_bs, T)

    Return 
        all_rewards: 
            array (n_trial, n_algos, rec_bs, T)
        all_items: 
            array (n_trial, n_algos, rec_bs, T)
    """
    root_path = os.path.join(args.root_proj_dir, 'results', args.dataset, algo_group, timestr, 'trial')
    load_item_dict = defaultdict(list)
    for algo_prefix in algo_prefixes:
        for load_item in ['rewards', 'items']:
            search_names = os.path.join(root_path, '{}-{}-{}-*'.format(trials, load_item, algo_prefix))
            print(search_names)
            filenames = glob.glob(search_names)
            if len(filenames) > 0:
                all_trial_rewards = []
                for filename in filenames:
                    print(filename)
                    rewards = np.load(filename)
                    rewards = np.expand_dims(rewards, axis = 0)[:,:,:,:T]
                    print(rewards.shape)
                    assert len(rewards.shape) == 4   
                    all_trial_rewards.append(rewards)
                
                all_trial_rewards = np.concatenate(all_trial_rewards, axis = 0)
                print('Collect trials {} for {}: {}'.format(load_item, algo_prefix, all_trial_rewards.shape))
                print('Debug n_trial {} rec_batch_size {}'.format(n_trial, rec_batch_size))

                if int(all_trial_rewards.shape[0]) >= n_trial and int(all_trial_rewards.shape[2]) == rec_batch_size: # only collect rewards with required trials
                    load_item_dict[load_item].append(all_trial_rewards[:n_trial,:,:,:])
                    if load_item == 'rewards':
                        algo_names.append(algo_prefix)          
            else:
                print('No file found for ', search_names)
        
        print('Debug len(load_item_dict[rewards]): ', len(load_item_dict['rewards']))
        print('Debug len(algo_names): ', len(algo_names))
        assert len(load_item_dict['rewards']) == len(algo_names)
    
    return load_item_dict['rewards'], load_item_dict['items'], algo_names

class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

def run_eva(args):
    algo_names = []
    all_rewards = []
    all_items = []
    
    trials = '[0-4]'
    T = 2000
    n_trial = 5
    # num_selected_users = 10

    algo_prefixes = []

    timestr = '20220507-0212'
    algo_group = 'test_twostage'
    n_inference = 1
    gamma = 0
    for algo in ['2_neuralgreedy']: # ['2_neuralucb']: # 
        for dynamic_aggregate_topic in [True, False]:
            algo_prefix = algo + '_ninf' + str(n_inference) + '_gamma' + str(gamma) + '_dynTopic' + str(dynamic_aggregate_topic)
            algo_prefixes.append(algo_prefix)

    eva_path = os.path.join(args.root_proj_dir, 'results', algo_group, timestr, 'eva')
    if not os.path.exists(eva_path):
        os.mkdir(eva_path) 
    log_path = os.path.join(eva_path, 'result_metrics.log')
    # log_file = open(log_path, 'w')
    sys.stdout = Logger(log_path)
    # sys.stdout = log_file

    all_rewards, all_items, algo_names = collect_rewards(args, algo_group, timestr, algo_prefixes, algo_names, all_rewards, all_items, trials, T, n_trial)
    
    all_rewards = np.concatenate(all_rewards, axis = 1) # (n_trial, n_algos, rec_bs, T)
    print('Collect all algos rewards: ', all_rewards.shape)

    metrics = cal_metric(all_rewards, algo_names, ['ctr', 'cumu_reward']) # , 'cumu_reward', 'ctr'
    plot_metrics(args, eva_path, metrics, algo_names, plot_title=str(n_trial) + ' trials', save_title = algo_group + '-' + timestr)

    # all_items = np.concatenate(all_items, axis = 1) # (n_trial, n_algos, rec_bs, T)
    # print('Collect all algos items: ', all_items.shape)
    # metrics = cal_diversity(args, all_items, algo_names)
    # plot_metrics(args, eva_path, metrics, algo_names, plot_title='One stage '+trials, save_title = algo_group + '-' + timestr)

if __name__ == '__main__':
    from configs.params import parse_args

    args = parse_args()
    # cal_base_ctr(args)
    run_eva(args)