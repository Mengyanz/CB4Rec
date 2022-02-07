"""Run evaluation. """

import math, os 
import numpy as np 
import matplotlib.pyplot as plt


def cal_metric(h_rewards_all, metric_names = ['cumu_reward']):
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
                print('Algorithm # ', i)
                print('Mean: ', cumu_rewards_mean[i][-1])
                print('Std: ', cumu_rewards_std[i][-1])

            metrics[metric] = [cumu_rewards_mean, cumu_rewards_std]

        if metric == 'ctr':
            ave_ctr = np.cumsum(np.mean(h_rewards_all, axis=2), axis = -1)/np.arange(1, T+1) # n_trails, n_algos, T
            ave_ctr_mean = np.mean(ave_ctr, axis = 0) # n_algos, T
            ave_ctr_std = np.std(ave_ctr, axis = 0) # n_algos, T

            for i in range(n_algos):
                print('Algorithm # ', i)
                print('Mean: ', ave_ctr_mean[i][-1])
                print('Std: ', ave_ctr_std[i][-1])

            metrics[metric] = [ave_ctr_mean, ave_ctr_std]
    
    return metrics

def plot_metrics(args, metrics, plot_title):
    plt_path = os.path.join(args.root_proj_dir, 'plots')
    if not os.path.exists(plt_path):
        os.mkdir(plt_path) 

    for name, value in metrics.items():
        plt.figure()
        mean, std = value
        n_algos, T = mean.shape
        for i in range(n_algos):
            plt.plot(range(T), mean[i], label = str(i))
            plt.fill_between(range(T), mean[i] + std[i], mean[i] - std[i], alpha = 0.2)
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel(name)
        plt.title(plot_title)
        plt.savefig(os.path.join(plt_path, name + '.png'))

def main():
    # from configs.thanh_params import parse_args
    from configs.mezhang_params import parse_args

    args = parse_args()
    result_path = os.path.join(args.root_proj_dir, "results", "rewards-0.npy")
    
    h_rewards_all = np.load(result_path)
    metrics = cal_metric(h_rewards_all, ['cumu_reward', 'ctr'])
    plot_metrics(args, metrics, plot_title=result_path)


if __name__ == '__main__':
    main()