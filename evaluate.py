"""Run evaluation. """

import math, os, sys
import numpy as np 
import matplotlib.pyplot as plt
from collections import defaultdict
import glob 
from utils.data_util import load_word2vec
import torch 
from torch.utils.data import DataLoader
from algorithms.nrms_sim import NRMS_IPS_Sim 
from utils.data_util import NewsDataset, TrainDataset, load_word2vec
from utils.metrics import ILAD, ILMD

def get_sim_news_embs(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    _, _, nindex2vec = load_word2vec(args, 'utils')
    news_dataset = NewsDataset(nindex2vec) 
    news_dl = DataLoader(news_dataset,batch_size=1024, shuffle=False, num_workers=2)

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
        print('Debug all_batch_ILADs shape: ', all_batch_ILADs.shape)
        print('Debug all_trial_ILADs shape: ', all_trial_ILADs.shape)
            
        batch_means.append(np.mean(all_batch_ILADs, axis = 0))
        batch_stds.append(np.std(all_batch_ILADs, axis = 0))
        trial_means.append(np.mean(all_trial_ILADs, axis = 0))
        trial_stds.append(np.std(all_trial_ILADs, axis = 0))

        print('Algorithm: ', algo_names[algo_i])
        print('ILAD-batch Mean: ', batch_means[algo_i].mean())
        print('ILAD-batch Std: ', batch_stds[algo_i].mean())
        print('ILAD-trial Mean: ', trial_means[algo_i][-1])
        print('ILAD-trial Std: ', trial_stds[algo_i][-1])

    metrics['ILAD-batch'] = [np.array(batch_means), np.array(batch_stds)]
    metrics['ILAD-trial'] = [np.array(trial_means), np.array(trial_stds)]
    return metrics


def cal_metric(h_rewards_all, algo_names, metric_names = ['cumu_reward']):  
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

def plot_metrics(args, eva_path, metrics, algo_names, plot_title, save_title = None):
    # plt_path = os.path.join(args.root_proj_dir, 'plots')
    

    for name, value in metrics.items():
        plt.figure()
        mean, std = value
        n_algos, T = mean.shape
        for i in range(n_algos):
            if name == 'raw_reward' or name == 'ave_ctr':
                # if i == 3:
                plt.scatter(range(T), mean[i], label = algo_names[i], s = 1, alpha = 0.5)
            elif 'ILAD' in name:
                x = list(range(T * 100 +100)[::100])[1:]
                plt.plot(x, mean[i], label = algo_names[i])
                plt.fill_between(x, mean[i] + std[i], mean[i] - std[i], alpha = 0.2)
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

def collect_rewards(args, algo_group, timestr, algo_prefixes, algo_names, all_rewards, all_items, trials = '[0-9]', T =2000):
    """collect rewards 
    the rewards files are saved in the results/algo_group/timestr/trial-algo_prefix-round.npy
    each file stores array with shape (n_algos, rec_bs, T)

    Return 
        all_rewards: 
            array (n_trial, n_algos, rec_bs, T)
        all_items: 
            array (n_trial, n_algos, rec_bs, T)
    """
    root_path = os.path.join(args.root_proj_dir, 'results', algo_group, timestr, 'trial')
    load_item_dict = defaultdict(list)
    for algo_prefix in algo_prefixes:
        algo_names.append(algo_prefix)
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

                print('Debug rewards.shape[0]', all_trial_rewards.shape[0])
                print('Debug trials[-2]:, ', trials[-2])
                if int(all_trial_rewards.shape[0]) == int(trials[-2]) + 1: # only collect rewards with required trials
                    load_item_dict[load_item].append(all_trial_rewards)
            
        else:
            print('No file found for ', search_names)
    
    return load_item_dict['rewards'], load_item_dict['items'], algo_names

def main(args):
    # plot_folder = '# user = 10' # 'tune_topic_update_period'
    # filenames=[]
    # filenames.append(os.path.join(args.root_proj_dir, "results", "linucb", "rewards-linucb-topicUpdate100-ninfernece5-dynamicFalse-0-2000.npy"))

    plot_folder = 'tune_topic_update_period'
    filenames = glob.glob(os.path.join(args.root_proj_dir, "results", plot_folder, "rewards-*-0-2000.npy"))
    filenames.append(os.path.join(args.root_proj_dir, "results", "2_ts_neuralucb", "rewards-2_ts_neuralucb-topicUpdate100-ninfernece5-dynamicFalse-9-2000.npy"))
    filenames.append(os.path.join(args.root_proj_dir, "results", "baselines", "rewards-baselines-greedy-8-2000.npy"))
    filenames.append(os.path.join(args.root_proj_dir, "results", "baselines", "rewards-baselines-linucb-7-2000.npy"))
    filenames.append(os.path.join(args.root_proj_dir, "results", "neural_dropoutucb", "rewards-neural_dropoutucb-topicUpdate100-ninfernece5-dynamicFalse-0-2000.npy"))
    filenames.append(os.path.join(args.root_proj_dir, "results", "tune_neural_linear",  "20220313-1415", "trial", "0-rewards-neural_glmadducb-2000.npy"))

    # plot_folder = 'Dynamic Topics'
    # filenames = glob.glob(os.path.join(args.root_proj_dir, "results", '2_ts_neuralucb_zhenyu', "rewards-*-1-2000.npy"))

    print('Debug filenames: ', filenames)
    algo_names = []
    all_rewards = []
    for filename in filenames:
        print(filename)

        if 'greedy' in filename:
            algo_name = 'neural_greedy'
        elif 'neural_glmadducb'in filename:
            algo_name = 'neural_glmadducb'
        elif 'linucb' in filename:
            algo_name = 'linucb'
        elif 'neural_dropoutucb' in filename:
            algo_name = 'neuralucb'
        elif '2_ts_neuralucb' in filename and 'zhenyu' not in filename:
            algo_name = 'ThompsonSampling_NeuralDropoutUCB_topicUpdate1'
        else:
            algo_name = ''.join(filename.split('-')[3:5])
        
            if '2_neuralucb_neuralucb' in algo_name:
                algo_name = algo_name.replace('2_neuralucb_neuralucb', '2_neuralucb_neuralucb_')
            if '2_ts_neuralucb_zhenyu' in algo_name:
                algo_name = algo_name.replace('2_ts_neuralucb_zhenyu', '2_neuralucb_neuralucb_')
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

class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
   
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
    # num_selected_users = 10

    algo_prefixes = []

    # timestr = '20220325-0242'
    # algo_group = 'test_dropout'
    # for algo in ['greedy', 'neural_dropoutucb', 'uniform_random']:
    #     algo_prefixes.append(algo)

    # timestr = '20220325-0535'
    # algo_group = 'test_lin_glm_neural_ucb'
    # for algo in ['linucb', 'glmucb', 'neural_linucb', 'neural_glmucb']:
    #     algo_prefixes.append(algo) 

    # timestr = '20220428-1453'
    # algo_group = 'test_lin_glm_neural_ucb'
    # for algo in ['glmucb']:
    #     algo_prefixes.append(algo) 
    
    # timestr = '20220502-0355'
    # algo_group = 'test_reload'
    # algo_prefixes.append('greedy_T400_reloadTrue')
    # algo_prefixes.append('greedy_T400_reloadFalse')
    # algo_prefixes.append('greedy_T200_reloadFalse')

    # timestr = '20220430-0000'
    # algo_group = 'run_onestage_nonneural'
    # for num_selected_users in [100, 1000]:
    #     for algo in ['uniform_random', 'linucb', 'glmucb']:
    #         algo_prefixes.append(algo + '_nuser' + str(num_selected_users))

    # timestr = '20220429-1242'
    # algo_group = 'run_onestage_neural'
    # for num_selected_users in [100, 1000]:
    #     for algo in ['greedy', 'neural_dropoutucb', 'neural_linucb', 'neural_glmucb', 'neural_gbilinucb', 'neural_glmadducb']:
    #         algo_prefixes.append(algo + '_nuser' + str(num_selected_users)) 

    timestr = '20220501-0546'
    algo_group = 'run_onestage_neural'
    for num_selected_users in [10, 100, 1000]:
        for glm_lr in [0.0001, 0.01]:
            for algo in ['neural_glmadducb', 'neural_gbilinucb']: # 'greedy', 'neural_dropoutucb', 'neural_linucb', 'neural_glmucb', 
                algo_prefixes.append(algo + '_nuser' + str(num_selected_users) + '_glmlr' + str(glm_lr))

    # timestr = '20220325-1500'
    # algo_group = 'test_proposed'
    # for algo in ['neural_gbilinucb']: # , 'neural_glmadducb'  
    #     for gamma in [0, 0.5]:
    #         algo_prefixes.append(algo + '-gamma' + str(gamma))
    # for algo in ['neural_glmadducb']: # , '  
    #     for gamma in [0.1, 0.5, 1]:
    #         algo_prefixes.append(algo + '-gamma' + str(gamma))

    # timestr = '20220405-0626'
    # algo_group = 'test_proposed'
    # for algo in ['neural_gbilinucb']: # , 'neural_glmadducb'  
    #     for gamma in [0, 0.1]:
    #         algo_prefixes.append(algo + '-gamma' + str(gamma))

    # timestr = '20220427-1434'
    # algo_group = 'debug_glm'
    # random_init = False
    # for algo in ['neural_glmucb']:  
    #     for glm_lr in [0.01, 0.1]:
    #         algo_prefixes.append(algo + '_randomInit' + str(random_init) + '_glmlr' + str(glm_lr))

    # timestr = '20220428-0228'
    # algo_group = 'test_proposed'
    # for algo in ['neural_gbilinucb', 'neural_glmadducb']: # 'neural_glmadducb', 
    #     for gamma in [0.1]: # 0, 0.1, 0.5, 1
    #         algo_prefixes.append(algo + '-gamma' + str(gamma))

    # timestr = '20220323-0738'
    # algo_group = 'tune_pretrainedMode_nuser'
    # for pretrained_mode in [True, False]:
    #     for num_selected_users in [10, 100, 1000]:
    #         reward_type = 'threshold_eps'
    #         algo = 'greedy'
    #         algo_prefixes.append(algo + '-pretrained' + str(pretrained_mode) + '-num_selected_users' + str(num_selected_users))
            

    # timestr = '20220322-0834'
    # algo_group = 'tune_pretrainedMode_rewardType'
    # for pretrained_mode in [True, False]:
    #     for reward_type in ['soft', 'hybrid', 'hard', 'threshold']:
    #         algo = 'greedy'
    #         algo_prefixes.append(algo + '-pretrained' + str(pretrained_mode) + '-reward' + str(reward_type))


    # timestr = '20220317-0327'
    # algo_group = 'single_stage'
    # num_selected_users = 10
    # for algo in ['neural_dropoutucb', 'greedy', 'linucb']:
    #     algo_prefixes.append(algo_group + '-' + algo )
    
    # timestr = '20220316-0643'
    # algo_group = 'test'
    # num_selected_users = 10
    # for algo in ['glmucb']:
    #         for epochs in [1, 5]:
    #             for lr in [0.1, 0.01, 0.001]:
    #                 algo_prefixes.append(algo + '-num_selected_users' + str(num_selected_users) + '-epochs' + str(epochs) + '-lr' + str(lr))
    # for algo in ['neural_dropoutucb']: # , 'greedy', 'linucb'
    #         algo_prefixes.append(algo_group + '-' + algo + '-num_selected_users' + str(num_selected_users))

    # timestr = '20220318-1322' #'20220316-0642'
    # algo_group = 'tune_neural_linear'
    # for algo in ['neural_glmadducb', 'NeuralGBiLinUCB']:
    #     for gamma in [0, 0.1]: #[0, 0.1, 0.5, 1]:
    #         algo_prefixes.append(algo + '-gamma' + str(gamma) + '-num_selected_users' + str(num_selected_users))

    # timestr = '20220319-0633' # '20220316-0643'
    # algo_group = 'tune_topic_update_period'
    # for algo in ['2_neuralucb_neuralucb', '2_ts_neuralucb']:
    #         if algo == '2_ts_neuralucb':
    #             updates = [1]
    #         else:
    #             updates = [10,50,100]
    #         for topic_update_period in updates:
    #             algo_prefixes.append(algo + '-topicUpdate' + str(topic_update_period) + '-num_selected_users' + str(num_selected_users))
    #             # algo_prefixes.append(algo_group + '-' + algo + '-topicUpdate' + str(topic_update_period))
    eva_path = os.path.join(args.root_proj_dir, 'results', algo_group, timestr, 'eva')
    if not os.path.exists(eva_path):
        os.mkdir(eva_path) 
    log_path = os.path.join(eva_path, 'result_metrics.log')
    # log_file = open(log_path, 'w')
    sys.stdout = Logger(log_path)
    # sys.stdout = log_file

    all_rewards, all_items, algo_names = collect_rewards(args, algo_group, timestr, algo_prefixes, algo_names, all_rewards, all_items, trials, T)
    
    # algo_group = 'tune_neural_linear'
    # algo_prefixes = ['neural_linearts', 'neural_linucb']

    # all_rewards = collect_rewards(args, algo_group, timestr, algo_prefixes, all_rewards, trials, T)
    # algo_names.extend(algo_prefixes)

    all_rewards = np.concatenate(all_rewards, axis = 1) # (n_trial, n_algos, rec_bs, T)
    print('Collect all algos rewards: ', all_rewards.shape)

    all_items = np.concatenate(all_items, axis = 1) # (n_trial, n_algos, rec_bs, T)
    print('Collect all algos items: ', all_items.shape)

    metrics = cal_metric(all_rewards, algo_names, ['ctr']) # , 'cumu_reward', 'ctr'
    plot_metrics(args, eva_path, metrics, algo_names, plot_title='One stage '+trials, save_title = algo_group + '-' + timestr)

    # metrics = cal_diversity(args, all_items, algo_names)
    # plot_metrics(args, eva_path, metrics, algo_names, plot_title='One stage '+trials, save_title = algo_group + '-' + timestr)

if __name__ == '__main__':
    # from configs.thanh_params import parse_args
    from configs.mezhang_params import parse_args
    # from configs.zhenyu_params import parse_args


    args = parse_args()
    # main(args)
    # cal_base_ctr(args)
    run_eva(args)