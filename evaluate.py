"""Run evaluation. """

import math, os 
import numpy as np 
import matplotlib.pyplot as plt
from collections import defaultdict
import glob 


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

def plot_metrics(args, metrics, algo_names, plot_title):
    plt_path = os.path.join(args.root_proj_dir, 'plots')
    if not os.path.exists(plt_path):
        os.mkdir(plt_path) 

    for name, value in metrics.items():
        plt.figure()
        mean, std = value
        n_algos, T = mean.shape
        for i in range(n_algos):
            if name == 'raw_reward' or name == 'ave_ctr':
                # if i == 3:
                plt.scatter(range(T), mean[i], label = algo_names[i], s = 1, alpha = 0.5)
            else:
                plt.plot(range(T), mean[i], label = algo_names[i])
                plt.fill_between(range(T), mean[i] + std[i], mean[i] - std[i], alpha = 0.2)
        
        if name == 'ctr' or name == 'ave_ctr':
            plt.ylim(0,1)
            plt.plot([0, 4999], [0.075, 0.075], label = 'uniform_random', color = 'grey', linestyle='--')
        plt.legend(bbox_to_anchor=(1.04,0.5), loc='center left')
        plt.xlabel('Iteration')
        plt.ylabel(name)
        plt.title(plot_title.replace('_', ' '))
        plt.savefig(os.path.join(plt_path, plot_title + '_' + name + '.pdf'),bbox_inches='tight')


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

def collect_rewards(args, algo_group, timestr, algo_prefixes, algo_names, all_rewards, trials = '[0-9]', T =2000):
    """collect rewards 
    the rewards files are saved in the results/algo_group/timestr/trial-algo_prefix-round.npy
    each file stores array with shape (n_algos, rec_bs, T)

    Return 
        all_rewards: 
            array (n_trial, n_algos, rec_bs, T)
    """
    root_path = os.path.join(args.root_proj_dir, 'results', algo_group, timestr, 'trial')
    for algo_prefix in algo_prefixes:
        search_names = os.path.join(root_path, '{}-rewards-{}-*'.format(trials, algo_prefix))
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
            print('Collect trials rewards for {}: {}'.format(algo_prefix, all_trial_rewards.shape))

            all_rewards.append(all_trial_rewards)
            algo_names.append(algo_prefix)
        else:
            print('No file found for ', search_names)
    
    return all_rewards, algo_names

def main(args):
    # plot_folder = '# user = 10' # 'tune_topic_update_period'
    # filenames=[]
    # filenames.append(os.path.join(args.root_proj_dir, "results", "single_linucb", "rewards-single_linucb-topicUpdate100-ninfernece5-dynamicFalse-0-2000.npy"))

    plot_folder = 'tune_topic_update_period'
    filenames = glob.glob(os.path.join(args.root_proj_dir, "results", plot_folder, "rewards-*-0-2000.npy"))
    filenames.append(os.path.join(args.root_proj_dir, "results", "ts_neuralucb", "rewards-ts_neuralucb-topicUpdate100-ninfernece5-dynamicFalse-9-2000.npy"))
    filenames.append(os.path.join(args.root_proj_dir, "results", "baselines", "rewards-baselines-greedy-8-2000.npy"))
    filenames.append(os.path.join(args.root_proj_dir, "results", "baselines", "rewards-baselines-single_linucb-7-2000.npy"))
    filenames.append(os.path.join(args.root_proj_dir, "results", "single_neuralucb", "rewards-single_neuralucb-topicUpdate100-ninfernece5-dynamicFalse-0-2000.npy"))
    filenames.append(os.path.join(args.root_proj_dir, "results", "tune_neural_linear",  "20220313-1415", "trial", "0-rewards-neuralglmucb_uihybrid-2000.npy"))

    # plot_folder = 'Dynamic Topics'
    # filenames = glob.glob(os.path.join(args.root_proj_dir, "results", 'ts_neuralucb_zhenyu', "rewards-*-1-2000.npy"))

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
    plot_metrics(args, metrics, algo_names, plot_title=plot_folder)

def run_eva(args):
    algo_names = []
    all_rewards = []
    
    trials = '[0]'
    T = 5000
    # num_selected_users = 10

    algo_prefixes = []

    timestr = '20220323-0738'
    algo_group = 'tune_pretrainedMode_nuser'
    for pretrained_mode in [True, False]:
        for num_selected_users in [10, 100, 1000]:
            reward_type = 'threshold-eps'
            algo = 'greedy'
            algo_prefixes.append(algo + '-pretrained' + str(pretrained_mode) + '-num_selected_users' + str(num_selected_users))
            

    # timestr = '20220322-0834'
    # algo_group = 'tune_pretrainedMode_rewardType'
    # for pretrained_mode in [True, False]:
    #     for reward_type in ['soft', 'hybrid', 'hard', 'threshold']:
    #         algo = 'greedy'
    #         algo_prefixes.append(algo + '-pretrained' + str(pretrained_mode) + '-reward' + str(reward_type))


    # timestr = '20220317-0327'
    # algo_group = 'single_stage'
    # num_selected_users = 10
    # for algo in ['single_neuralucb', 'greedy', 'single_linucb']:
    #     algo_prefixes.append(algo_group + '-' + algo )
    
    # timestr = '20220316-0643'
    # algo_group = 'test'
    # num_selected_users = 10
    # for algo in ['glmucb']:
    #         for epochs in [1, 5]:
    #             for lr in [0.1, 0.01, 0.001]:
    #                 algo_prefixes.append(algo + '-num_selected_users' + str(num_selected_users) + '-epochs' + str(epochs) + '-lr' + str(lr))
    # for algo in ['single_neuralucb']: # , 'greedy', 'single_linucb'
    #         algo_prefixes.append(algo_group + '-' + algo + '-num_selected_users' + str(num_selected_users))

    # timestr = '20220318-1322' #'20220316-0642'
    # algo_group = 'tune_neural_linear'
    # for algo in ['neuralglmucb_uihybrid', 'neuralbilinucb_hybrid']:
    #     for gamma in [0, 0.1]: #[0, 0.1, 0.5, 1]:
    #         algo_prefixes.append(algo + '-gamma' + str(gamma) + '-num_selected_users' + str(num_selected_users))

    # timestr = '20220319-0633' # '20220316-0643'
    # algo_group = 'tune_topic_update_period'
    # for algo in ['neuralucb_neuralucb', 'ts_neuralucb']:
    #         if algo == 'ts_neuralucb':
    #             updates = [1]
    #         else:
    #             updates = [10,50,100]
    #         for topic_update_period in updates:
    #             algo_prefixes.append(algo + '-topicUpdate' + str(topic_update_period) + '-num_selected_users' + str(num_selected_users))
    #             # algo_prefixes.append(algo_group + '-' + algo + '-topicUpdate' + str(topic_update_period))

    all_rewards, algo_names = collect_rewards(args, algo_group, timestr, algo_prefixes, algo_names, all_rewards, trials, T)
    
    # algo_group = 'tune_neural_linear'
    # algo_prefixes = ['neural_linearts', 'neural_linearucb']

    # all_rewards = collect_rewards(args, algo_group, timestr, algo_prefixes, all_rewards, trials, T)
    # algo_names.extend(algo_prefixes)

    all_rewards = np.concatenate(all_rewards, axis = 1) # (n_trial, n_algos, rec_bs, T)
    print('Collect all algos rewards: ', all_rewards.shape)

    metrics = cal_metric(all_rewards, algo_names, ['ctr']) # , 'cumu_reward', 'ctr'
    plot_metrics(args, metrics, algo_names, plot_title='IPS-Threshold-eps0.1-' + algo_group+trials)


if __name__ == '__main__':
    # from configs.thanh_params import parse_args
    from configs.mezhang_params import parse_args
    # from configs.zhenyu_params import parse_args

    args = parse_args()
    # main(args)
    # cal_base_ctr(args)
    run_eva(args)