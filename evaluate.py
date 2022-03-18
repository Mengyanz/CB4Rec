"""Run evaluation. """

import math, os 
import numpy as np 
import matplotlib.pyplot as plt
from collections import defaultdict
import glob 
from tqdm import tqdm

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def cal_item_metric(rec_items_all, algo_names):
    nindex2embedding = np.load("./data/large/utils/nindex2embedding.npy")
    n_trials, n_algos, rec_bs, T = rec_items_all.shape
    print('# trails: ', n_trials, ', # algos: ', n_algos, ', # rec_bs: ', rec_bs, ', T: ', T)
    rec_items_all = rec_items_all.transpose(1, 0, 3, 2) # n_algos, n_trials, T, rec_bs
    metrics = {}
    means = []
    stds = []
    for algo_i in tqdm(range(rec_items_all.shape[0]), total=rec_items_all.shape[0]):
        scores = []
        for trial_i in range(rec_items_all.shape[1]):
            rec_items = rec_items_all[algo_i][trial_i]
            all_embedding = []
            for batch in rec_items.T:
                batch = list(map(int, batch))
                all_embedding.append(nindex2embedding[batch])
            all_embedding = np.concatenate(all_embedding, axis=0)
            
            for i in tqdm(range(all_embedding.shape[0])):
                for j in range(all_embedding.shape[1]):
                    if i == j:
                        continue
                    scores.append(1.0/2 - sigmoid(all_embedding[i] @ all_embedding[j])/2)
        
        metrics[algo_names[algo_i]] = [np.mean(scores), np.std(scores)]
    return metrics
        
    
        

def cal_metric(h_rewards_all, algo_names, metric_names = ['cumu_reward'],):  
    n_trials, n_algos, rec_bs, T = h_rewards_all.shape
    print('# trails: ', n_trials, ', # algos: ', n_algos, ', # rec_bs: ', rec_bs, ', T: ', T)
    metrics = {}
    
    for metric in metric_names:
        print('Metric: ', metric)
        if metric == 'cumu_reward':
            cumu_rewards = np.cumsum(np.sum(h_rewards_all, axis=2), axis = -1) # n_trails, n_algos, T
            cumu_rewards_mean = np.mean(cumu_rewards, axis = 0) # n_algos, T
            cumu_rewards_std = np.std(cumu_rewards, axis = 0) # n_algos, T

            for i in range(n_algos):
                print('Algorithm: ', algo_names[i])
                print('Mean: ', cumu_rewards_mean[i][-1])
                print('Std: ', cumu_rewards_std[i][-1])

            metrics[metric] = [cumu_rewards_mean, cumu_rewards_std]

        if metric == 'ctr':
            ave_ctr = np.cumsum(np.mean(h_rewards_all, axis=2), axis = -1)/np.arange(1, T+1) # n_trails, n_algos, T
            ave_ctr_mean = np.mean(ave_ctr, axis = 0) # n_algos, T
            ave_ctr_std = np.std(ave_ctr, axis = 0) # n_algos, T

            for i in range(n_algos):
                print('Algorithm: ', algo_names[i])
                print('Mean: ', ave_ctr_mean[i][-1])
                print('Std: ', ave_ctr_std[i][-1])

            metrics[metric] = [ave_ctr_mean, ave_ctr_std]

        if metric == 'ave_ctr':
            ave_ctr = np.mean(h_rewards_all, axis=2)
            ave_ctr_mean = np.mean(ave_ctr, axis = 0) # n_algos, T
            ave_ctr_std = np.std(ave_ctr, axis = 0) # n_algos, T

            for i in range(n_algos):
                print('Algorithm: ', algo_names[i])
                print('Mean: ', ave_ctr_mean[i][-1])
                print('Std: ', ave_ctr_std[i][-1])

            metrics[metric] = [ave_ctr_mean, ave_ctr_std]


        if metric == 'raw_reward':
            cumu_rewards = np.sum(h_rewards_all, axis=2)
            cumu_rewards_mean = np.mean(cumu_rewards, axis = 0) # n_algos, T
            cumu_rewards_std = np.std(cumu_rewards, axis = 0) # n_algos, T

            for i in range(n_algos):
                print('Algorithm: ', algo_names[i])
                print('Mean: ', cumu_rewards_mean[i][-1])
                print('Std: ', cumu_rewards_std[i][-1])

            metrics[metric] = [cumu_rewards_mean, cumu_rewards_std]
    
    return metrics

def plot_item_metrics(args, metrics, algo_names, plot_title):
    plt_path = os.path.join(args.root_proj_dir, 'plots')
    if not os.path.exists(plt_path):
        os.mkdir(plt_path) 
    plt.figure()
    i = 0
    for name, value in metrics.items():
        mean, std = value
        # n_algos, T = mean.shape
        # for i in range(n_algos):
        plt.bar(i, mean, label = algo_names[i][:15])
        # plt.fill_between(i, mean+std, mean - std, alpha = 0.2)
        plt.legend()
        i += 1
    plt.xlabel('algorithms')
    plt.ylabel('diversity score')
    plt.title(plot_title)
    plt.savefig(os.path.join(plt_path, plot_title + '.png'))

def plot_metrics(args, metrics, algo_names, plot_title):
    plt_path = os.path.join(args.root_proj_dir, 'plots')
    if not os.path.exists(plt_path):
        os.mkdir(plt_path) 

    for name, value in metrics.items():
        plt.figure()
        mean, std = value
        n_algos, T = mean.shape
        for i in range(n_algos):
            if name == 'raw_reward' or 'ave_ctr':
                # if i == 3:
                plt.scatter(range(T), mean[i], label = algo_names[i], s = 1, alpha = 0.5)
            else:
                plt.plot(range(T), mean[i], label = algo_names[i])
                plt.fill_between(range(T), mean[i] + std[i], mean[i] - std[i], alpha = 0.2)
        
        if name == 'ctr' or name == 'ave_ctr':
            plt.ylim(0,1)
            plt.plot([0, 1999], [0.075, 0.075], label = 'uniform_random', color = 'grey', linestyle='--')
        plt.legend(bbox_to_anchor=(1.04,0.5), loc='center left')
        plt.xlabel('Iteration')
        plt.ylabel(name)
        plt.title(plot_title.replace('_', ' '))
        plt.savefig(os.path.join(plt_path, plot_title + '_' + name + '.pdf'),bbox_inches='tight')


def cal_base_ctr(args):
    """calculate base ctr: uniform random policy 10 trails average ctr
    """
    filename = 'rewards-uniform_random-0-1000.npy'
    h_rewards_all = np.array(np.load(os.path.join(args.root_proj_dir, "results", filename)))
    print(h_rewards_all.shape)
    all_rewards = np.concatenate([h_rewards_all], axis = 1)
    metrics = cal_metric(all_rewards, 'uniform_random', ['cumu_reward', 'ctr'])
    plot_metrics(args, metrics, 'uniform_random', plot_title='Uniform Random 0 trials')
    return metrics

def collect_rewards(args, algo_group, timestr, algo_prefixes, all_rewards, trials = '[0-9]', T =2000):
    """collect rewards 
    the rewards files are saved in the results/algo_group/timestr/trial-algo_prefix-round.npy
    each file stores array with shape (n_algos, rec_bs, T)

    Return 
        all_rewards: 
            array (n_trial, n_algos, rec_bs, T)
    """
    root_path = os.path.join(args.root_proj_dir, 'results', algo_group, timestr, 'trial')
    for algo_prefix in algo_prefixes:
        print(os.path.join(root_path, '{}-rewards-{}-*'.format(trials, algo_prefix)))
        filenames = glob.glob(os.path.join(root_path, '{}-rewards-{}-*'.format(trials, algo_prefix)))
        all_trial_rewards = []
        for filename in filenames:
            print(filename)
            rewards = np.load(filename)
            rewards = np.expand_dims(rewards, axis = 0)[:,:,:,:T]
            print(rewards.shape)
            assert len(rewards.shape) == 4
            all_trial_rewards.append(rewards)
        
        all_trial_rewards = np.concatenate(all_trial_rewards, axis = 0)
        print('Collect trials rewards for {}: {}'.format(algo_prefix, all_trial_rewards.shape))

        all_rewards.append(all_trial_rewards )
    
    return all_rewards

def main(args):
    filenames = glob.glob(os.path.join(args.root_proj_dir, "results","hcb", "rewards-*dynamicFalse*50-0-1000.npy")) + glob.glob(os.path.join(args.root_proj_dir, "results","phcb", "rewards-*dynamicFalse*50-0-1000.npy"))
    print('Debug filenames: ', filenames)
    algo_names = []
    all_rewards = []
    for filename in filenames:
        print(filename)

        if 'greedy' in filename:
            algo_name = 'neural_greedy'
        elif 'neuralglmucb_uihybrid'in filename:
            algo_name = 'neuralglmucb_uihybrid'
        elif 'single_linucb' in filename:
            algo_name = 'linucb'
        elif 'single_neuralucb' in filename:
            algo_name = 'neuralucb'
        elif 'ts_neuralucb' in filename and 'zhenyu' not in filename:
            algo_name = 'ThompsonSampling_neuralucb_topicUpdate1'
        else:
            algo_name = ''.join(filename.split('-')[3:5])
        
            if 'neuralucb_neuralucb' in algo_name:
                algo_name = algo_name.replace('neuralucb_neuralucb', 'neuralucb_neuralucb_')
            if 'ts_neuralucb_zhenyu' in algo_name:
                algo_name = algo_name.replace('ts_neuralucb_zhenyu', 'neuralucb_neuralucb_')
        algo_names.append(algo_name)
        h_rewards_all = np.load(filename)
        if len(h_rewards_all.shape) == 3: # TODO: remove after the save format is consistent
            h_rewards_all = np.expand_dims(h_rewards_all, axis = 0)
        h_rewards_all = h_rewards_all[0,:,:,:]

        if len(h_rewards_all.shape) == 3: # TODO: remove after the save format is consistent
            h_rewards_all = np.expand_dims(h_rewards_all, axis = 0)
        print(h_rewards_all.shape)
        all_rewards.append(h_rewards_all)
    all_rewards = np.concatenate(all_rewards, axis = 1)
    print(all_rewards.shape)
    
    metrics = cal_metric(all_rewards, algo_names, ['cumu_reward', 'ctr']) # , 'cumu_reward', 'ctr'
    plot_metrics(args, metrics, algo_names, plot_title='trail0')
    
    
    item_filenames = glob.glob(os.path.join(args.root_proj_dir, "results", "items-*dynamicFalse*-0-1000.npy"))
    print('Debug item_filenames: ', item_filenames)
    algo_names = []
    all_rewards = []
    for filename in item_filenames:
        print(filename)
        algo_name = '-'.join(filename.split('-')[1:6])
        algo_names.append(algo_name)
        h_rewards_all = np.load(filename)[:,:,:,:]
        if len(h_rewards_all.shape) == 3: # TODO: remove after the save format is consistent
            h_rewards_all = np.expand_dims(h_rewards_all, axis = 0)
        print(h_rewards_all.shape)
        all_rewards.append(h_rewards_all)
    all_rewards = np.concatenate(all_rewards, axis = 1)
    print(all_rewards.shape)
    
    metrics = cal_item_metric(all_rewards, algo_names) # , 'cumu_reward', 'ctr'
    plot_item_metrics(args, metrics, algo_names, plot_title='trail0_item')
    
if __name__ == '__main__':
    # from configs.thanh_params import parse_args
    #f rom configs.mezhang_params import parse_args
    from configs.zhenyu_params import parse_args

    args = parse_args()
    # main(args)
    # cal_base_ctr(args)
    # run_eva(args)