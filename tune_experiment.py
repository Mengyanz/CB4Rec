import os,sys,math
import subprocess
import time
import argparse
import random
from tkinter import FALSE
import numpy as np
import time

import matplotlib.pyplot as plt
from collections import defaultdict
import glob 
from utils.data_util import load_word2vec
import torch 
from torch.utils.data import DataLoader
from algorithms.nrms_sim import NRMS_IPS_Sim 
from utils.data_util import NewsDataset, TrainDataset, load_word2vec
from utils.metrics import ILAD, ILMD
from evaluate import Logger, cal_diversity, cal_metric, plot_metrics, collect_rewards

def multi_gpu_launcher(commands,gpus,models_per_gpu):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    procs = [None]*len(gpus)*models_per_gpu

    while len(commands) > 0:
        for i,proc in enumerate(procs):
            gpu_idx = gpus[i % len(gpus)]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this index; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs[i] = new_proc
                print("on GPU {} running {}".format(gpu_idx, cmd))
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs:
        if p is not None:
            p.wait()

def run_exps(args, algo_groups, result_path, gpus,models_per_gpu, timestr, simulate_flag=True, eva_flag=True, rec_batch_sizes=[5]):
    all_commands = []
    all_algo_prefixes = {}
    for algo_group in algo_groups:
        commands, algo_prefixes = create_commands(args, algo_group, result_path)
        all_commands += commands
        all_algo_prefixes[algo_group] = algo_prefixes
    # random.shuffle(commands)
    if simulate_flag:
        multi_gpu_launcher(all_commands,gpus,models_per_gpu)
    if eva_flag:
        for algo_group, algo_prefixes in all_algo_prefixes.items():
            eva(args, algo_group, timestr, algo_prefixes, rec_batch_sizes)

def eva(args, algo_group, timestr, algo_prefixes, rec_batch_sizes = [5]):
    # trials = '[0-{}]'.format(args.n_trials-1)
    trials = '[0-4]'
    T = args.T
    n_trial = args.n_trials 

    eva_path = os.path.join(args.root_proj_dir, 'results', algo_group, timestr, 'eva')
    if not os.path.exists(eva_path):
        os.mkdir(eva_path) 
    log_path = os.path.join(eva_path, 'result_metrics.log')
    # log_file = open(log_path, 'w')
    sys.stdout = Logger(log_path)
    # sys.stdout = log_file

    for rec_batch_size in rec_batch_sizes:
        print('Debug in eva: rec_batch_size: ', rec_batch_size)
        algo_names = []
        all_rewards = []
        all_items = []

        all_rewards, all_items, algo_names = collect_rewards(args, algo_group, timestr, algo_prefixes, algo_names, all_rewards, all_items, trials, T, n_trial, rec_batch_size)
        
        if len(algo_names) > 0:

            all_rewards = np.concatenate(all_rewards, axis = 1) # (n_trial, n_algos, rec_bs, T)
            print('==============Evaluate in {} ============='.format(eva_path))
            print('Collected all algos rewards: ', all_rewards.shape)

            all_items = np.concatenate(all_items, axis = 1) # (n_trial, n_algos, rec_bs, T)
            print('Collected all algos items: ', all_items.shape)

            metrics = cal_metric(all_rewards, algo_names, ['ctr', 'cumu_ctr', 'moving_ctr']) # , 'cumu_reward', 'ctr'
            plot_metrics(args, eva_path, metrics, algo_names, plot_title= trials)

            # metrics = cal_diversity(args, all_items, algo_names)
            # plot_metrics(args, eva_path, metrics, algo_names, plot_title= trials)
    print('{} {} evaluation Done!'.format(algo_group, timestr))

def create_commands(args, algo_group, result_path):
    commands = []
    algo_prefixes = []
    if algo_group == 'test_onestage':
        for num_selected_users in [10, 100, 1000]: 
            for algo in ['uniform_random', 'glmucb', 'greedy', 'neural_dropoutucb', 'neural_glmucb', 'neural_glmadducb', 'neural_gbilinucb']: 
                algo_prefix = algo + '_nuser' + str(num_selected_users) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {} --root_dir {} --algo_prefix {} --result_path {} --num_selected_users {} > {}".format(algo, args.root_dir, algo_prefix, result_path, num_selected_users, log_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'test_twostage':
        for algo in ['uniform_random', 'greedy', 'neural_dropoutucb', 'neural_glmadducb', 'neural_gbilinucb', '2_random','2_neuralgreedy', '2_neuralucb', '2_neuralglmadducb', '2_neuralglmbilinucb']: 
            for rec_batch_size in [1,5,10]: 
                algo_prefix = algo + '_recSize' + str(rec_batch_size) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {} --root_dir {}  --algo_prefix {} --result_path {} --rec_batch_size {}  > {}".format(algo, args.root_dir,  algo_prefix, result_path, rec_batch_size,log_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'test_dynamic_topic':
        for algo in ['2_neuralglmadducb', '2_neuralglmbilinucb']:
            for dynamic_aggregate_topic in [True, False]:
                algo_prefix = algo + '_dynTopic' + str(dynamic_aggregate_topic) #+ '_glmlr' + str(glm_lr)
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {} --root_dir {}  --algo_prefix {} --result_path {} --dynamic_aggregate_topic {} > {}".format(algo, args.root_dir, algo_prefix, result_path, dynamic_aggregate_topic, log_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'tune_gamma':
        for gamma in [0.01, 0.05, 0.1, 0.5, 1, 2]:
            for algo in ['2_neuralglmadducb', '2_neuralglmbilinucb']:
                algo_prefix = algo + '_gamma' + str(gamma)
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {} --root_dir {}  --algo_prefix {} --result_path {}  --gamma {} > {}".format(algo, args.root_dir, algo_prefix, result_path, gamma, log_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'test_largeT':
        T = 10000
        for algo in ['neural_glmadducb', 'neural_gbilinucb', '2_neuralglmadducb', '2_neuralglmbilinucb']:
            algo_prefix = algo + '_T' + str(T)
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {} --root_dir {}  --algo_prefix {} --result_path {}  --T {} > {}".format(algo, args.root_dir, algo_prefix, result_path, T, log_path))
            algo_prefixes.append(algo_prefix)
    else:
        raise NotImplementedError("No algo_group specified.")
    return commands, algo_prefixes

if __name__ == '__main__':
    from configs.params import parse_args
    args = parse_args()

    # settings
    gpus = [1]
    models_per_gpu = 2
    algo_groups = ['test_onestage', 'test_twostage', 'test_dynamic_topic', 'tune_gamma', 'test_largeT']
    args.root_dir = '/data4/u6015325/'
    args.root_data_dir = os.path.join(args.root_dir, args.root_data_dir)
    args.root_proj_dir = os.path.join(args.root_dir, args.root_proj_dir)
    args.result_path = os.path.join(args.root_dir, args.result_path)
    # args.n_trials = 1
    # args.T = 10000   
    simulate_flag=True
    rec_batch_size=[5] 
    timestr = time.strftime("%Y%m%d-%H%M")
    # timestr = '20220520-1418'
    
    print("============================algo groups: {} ==============================".format(algo_groups))
    print('Saving to {}'.format(timestr))
    for algo_group in algo_groups:
        result_path = os.path.join(args.root_proj_dir, 'results', algo_group, timestr)
        if not os.path.exists(result_path):
            os.makedirs(result_path) 
        trial_path = os.path.join(result_path, 'trial') # store final results
        if not os.path.exists(trial_path):
            os.mkdir(trial_path) 
        model_path = os.path.join(result_path, 'model') # store models (for future reload)
        if not os.path.exists(model_path):
            os.mkdir(model_path) 
   
    run_exps(args, algo_groups, result_path,gpus,models_per_gpu,timestr, simulate_flag=simulate_flag, eva_flag=True, rec_batch_sizes=rec_batch_size)