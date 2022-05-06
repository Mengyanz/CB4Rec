# Part of code from https://github.com/thanhnguyentang/offline_neural_bandits/blob/main/tune_realworld.py
import os
import subprocess
import time
import argparse
import random
import numpy as np
import time
from evaluate import *

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

def run_exps(args, algo_groups, result_path, gpus,models_per_gpu, timestr):
    all_commands = []
    all_algo_prefixes = {}
    for algo_group in algo_groups:
        commands, algo_prefixes = create_commands(args, algo_group, result_path)
        all_commands += commands
        all_algo_prefixes[algo_group] = algo_prefixes
    # random.shuffle(commands)
    multi_gpu_launcher(all_commands,gpus,models_per_gpu)
    for algo_group, algo_prefixes in all_algo_prefixes.items():
        eva(args, algo_group, timestr, algo_prefixes)

def eva(args, algo_group, timestr, algo_prefixes, ):
    algo_names = []
    all_rewards = []
    all_items = []

    trials = '[0-{}]'.format(args.n_trials)
    T = args.T
    n_trial = args.n_trials + 1

    eva_path = os.path.join(args.root_proj_dir, 'results', algo_group, timestr, 'eva')
    if not os.path.exists(eva_path):
        os.mkdir(eva_path) 
    log_path = os.path.join(eva_path, 'result_metrics.log')
    # log_file = open(log_path, 'w')
    sys.stdout = Logger(log_path)
    # sys.stdout = log_file

    all_rewards, all_items, algo_names = collect_rewards(args, algo_group, timestr, algo_prefixes, algo_names, all_rewards, all_items, trials, T, n_trial)
    
    # algo_group = 'tune_neural_linear'
    # algo_prefixes = ['neural_linearts', 'neural_linucb']

    # all_rewards = collect_rewards(args, algo_group, timestr, algo_prefixes, all_rewards, trials, T)
    # algo_names.extend(algo_prefixes)

    all_rewards = np.concatenate(all_rewards, axis = 1) # (n_trial, n_algos, rec_bs, T)
    print('==============Evaluate in {} ============='.format(eva))
    print('Collect all algos rewards: ', all_rewards.shape)

    all_items = np.concatenate(all_items, axis = 1) # (n_trial, n_algos, rec_bs, T)
    print('Collect all algos items: ', all_items.shape)

    metrics = cal_metric(all_rewards, algo_names, ['ctr', 'cumu_reward']) # , 'cumu_reward', 'ctr'
    plot_metrics(args, eva_path, metrics, algo_names, plot_title= trials)

    metrics = cal_diversity(args, all_items, algo_names)
    plot_metrics(args, eva_path, metrics, algo_names, plot_title= trials)


def create_commands(args, algo_group, result_path):
    commands = []
    algo_prefixes = []
    # num_selected_users = 10
    if algo_group == 'test_reset_buffer':
        algo = 'greedy'
        for reset in [True, False]:
            algo_prefix = algo + '_resetbuffer' + str(reset)
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --reset_buffer {}  > {}".format(algo, algo_prefix, result_path, reset, log_path))
            algo_prefixes.append(algo_prefix)

    elif algo_group == 'tune_glm':
        for algo in ['neural_glmucb']:
            for num_selected_users in [100, 1000]:
                for glm_lr in [1e-1,1e-2,1e-3,1e-4]:
                    algo_prefix = algo + '_glmlr' + str(glm_lr) + '_nuser' + str(num_selected_users) #+ 'score_budget' + str(score_budget)
                    # + '-' + str(args.n_trials) + '-' + str(args.T) 
                    log_path = os.path.join(result_path, algo_prefix + '.log')
                    commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --glm_lr {} > {}".format(algo, algo_prefix, result_path, num_selected_users, glm_lr, log_path))
                    algo_prefixes.append(algo_prefix)

    elif algo_group == 'test_twostage':
        n_inference = 1
        gamma = 0
        for algo in ['2_neuralgreedy_neuralgreedy']: # ['2_neuralucb_neuralucb']: # 
            for dynamic_aggregate_topic in [True, False]:
                algo_prefix = algo + '_ninf' + str(n_inference) + '_gamma' + str(gamma) + '_dynTopic' + str(dynamic_aggregate_topic)
                log_path = os.path.join(result_path, algo_prefix + '.log')
                # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} > {}".format(algo, algo_prefix, result_path, log_path))
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --n_inference {} --gamma {} --dynamic_aggregate_topic {} > {}".format(algo, algo_prefix, result_path, n_inference, gamma, dynamic_aggregate_topic, log_path))
                algo_prefixes.append(algo_prefix)
        
        # for algo in ['2_random']: # ['2_neuralucb_neuralucb']: # 
        #     algo_prefix = algo 
        #     log_path = os.path.join(result_path, algo_prefix + '.log')
        #     # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} > {}".format(algo, algo_prefix, result_path, log_path))
        #     commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {}  > {}".format(algo, algo_prefix, result_path, log_path))

    elif algo_group == 'debug_decrease_after_100':
        # 100 is the firs nrms update, try not to update
        algo = 'neural_glmadducb'
        n_trial = 10
        for update_period in [100, 3000]:
            algo_prefix = algo + '_update' + str(update_period)
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --update_period {} --n_trial {} > {}".format(algo, algo_prefix, result_path, update_period, n_trial, log_path))
            algo_prefixes.append(algo_prefix)

    elif algo_group == 'test_reload':
        algo = 'greedy'
        n_trial = 5
        # reward_type = 'threshold' # remove stochastic to check reload 

        # reload_flag = True
        # reload_path = os.path.join(args.root_proj_dir, 'results', 'test_reload', '20220502-0355', 'model', 'greedy_T200_reloadFalse-200')
        reload_flag = False
        reload_path = None

        for T in [2000]:
            algo_prefix = algo + '_T' + str(T) + '_reload' + str(reload_flag)
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --T {} --reload_flag {} --reload_path {}  --n_trial {} > {}".format(algo, algo_prefix, result_path, T, reload_flag, reload_path, n_trial, log_path))
            algo_prefixes.append(algo_prefix)
    elif algo_group == 'test_save_load':
        T = 500
        for algo in ['greedy']:
            algo_prefix = algo 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --T {} > {}".format(algo, algo_prefix, result_path, T, log_path))
            algo_prefixes.append(algo_prefix)
    elif algo_group == 'run_onestage_nonneural':
        for num_selected_users in [100, 1000]:
            for algo in ['uniform_random', 'linucb', 'glmucb']:
                algo_prefix = algo + '_nuser' + str(num_selected_users) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, num_selected_users, log_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'run_onestage_neural':
        T = 5000
        for num_selected_users in [10]: #  100, 1000
            # for glm_lr in [1e-3,1e-4]: # 0.0001, 0.01
            for algo in ['neural_gbilinucb', 'neural_glmadducb']: # 'greedy', 'neural_dropoutucb', 'neural_linucb', 'neural_glmucb', 'neural_glmadducb', 
                if algo == 'neural_gbilinucb':
                    glm_lr = 1e-3
                if algo == 'neural_glmadducb':
                    glm_lr = 0.01
                algo_prefix = algo + '_nuser' + str(num_selected_users) + '_glmlr' + str(glm_lr) + '_T' + str(T)
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --glm_lr {} --T {} > {}".format(algo, algo_prefix, result_path, num_selected_users, glm_lr, T, log_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'tune_dropout':
        algo = 'neural_dropoutucb'
        for num_selected_users in [100]: #  100, 1000
            for gamma in [0.2, 0.05]:
                algo_prefix = algo + '_nuser' + str(num_selected_users) + '_gamma' + str(gamma)
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --gamma {} > {}".format(algo, algo_prefix, result_path, num_selected_users, gamma, log_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'debug_glm':
        # T = 100
        # score_budget = 20
        for algo in ['neural_glmucb']: # ['neural_glmucb_lbfgs']: # ['neural_glmucb', 'neural_linucb']: 
            for glm_lr in [0.1, 0.01]:
                algo_prefix = algo + '_randomInit' + str(random_init) + '_glmlr' + str(glm_lr) #+ 'score_budget' + str(score_budget)
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --random_init {} --glm_lr {} > {}".format(algo, algo_prefix, result_path, num_selected_users, random_init, glm_lr, log_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'test_proposed':
        for algo in ['neural_gbilinucb', 'neural_glmadducb']: # 'neural_glmadducb', 
            for gamma in [0.1]: # 0, 0.1, 0.5, 1
                algo_prefix = algo + '-gamma' + str(gamma)
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --gamma {} > {}".format(algo, algo_prefix, result_path, num_selected_users, gamma, log_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'test_dropout':
        for algo in ['greedy', 'neural_dropoutucb', 'uniform_random']:
            algo_prefix = algo 
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, num_selected_users, log_path))
            algo_prefixes.append(algo_prefix)

    elif algo_group == 'tune_neural_linear':
        for algo in ['NeuralGBiLinUCB', 'neural_glmadducb']: # 'neural_linearts', 'neural_glmadducb'
            for gamma in [0, 0.1]:
                algo_prefix = algo  + '-gamma' + str(gamma) + '-num_selected_users' + str(num_selected_users)
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --gamma {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, gamma, num_selected_users, log_path))
                # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {}".format(algo, algo_prefix, result_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'tune_dynamicTopic':
        for algo in ['2_ts_neuralucb', '2_neuralucb_neuralucb']:
            for dynamic_aggregate_topic in [True]: # , False
                algo_prefix = algo + '-dynamicTopic' + str(dynamic_aggregate_topic) 
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --dynamic_aggregate_topic {}> {}".format(algo, algo_prefix, result_path, dynamic_aggregate_topic, log_path))
                algo_prefixes.append(algo_prefix)

    elif algo_group == 'tune_topic_update_period':
        for algo in ['2_neuralucb_neuralucb', '2_ts_neuralucb']:
            if algo == '2_ts_neuralucb':
                updates = [1]
            else:
                updates = [10,50,100]
            for topic_update_period in updates:
                algo_prefix = algo + '-topicUpdate' + str(topic_update_period) + '-num_selected_users' + str(num_selected_users)
                print('Debug algo_prefix: ', algo_prefix)
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --topic_update_period {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path,  topic_update_period, num_selected_users, log_path))
                algo_prefixes.append(algo_prefix)
    elif algo_group == 'single_stage':
        for algo in ['neural_dropoutucb', 'greedy', 'linucb', 'glmucb']: # , 'greedy', 'linucb'
            algo_prefix = algo 
            print('Debug algo_prefix: ', algo_prefix)
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} > {}".format(algo, algo_prefix, result_path,  log_path))
            algo_prefixes.append(algo_prefix)
    else:
        algo_prefix = algo_group +'-ninference' + str(args.n_inference) + '-dynamic' + str(args.dynamic_aggregate_topic)+'-splitlarge' + str(args.split_large_topic) +'-perRecScoreBudget' + str(args.per_rec_score_budget) 
        # + '-' + str(args.n_trials) + '-' + str(args.T) 
        log_path = os.path.join(result_path, algo_prefix + '.log')
        commands.append("python run_experiment.py --algo {} --algo_prefix {} --result_path {} > {}".format(algo_group, algo_prefix, result_path,  log_path))
        algo_prefixes.append(algo_prefix)
    return commands, algo_prefixes

if __name__ == '__main__':
    from configs.mezhang_params import parse_args
    # from configs.zhenyu_params import parse_args
    args = parse_args()
    os.system('ulimit -n 4096')

    # settings
    gpus = [1]
    models_per_gpu = 2
    algo_groups = ['test_reset_buffer']
    # algo_groups = ['test_twostage']
    # algo_groups = ['tune_dropout']
    # algo_groups =  ['run_onestage_neural'] 
    # algo_groups = ['tune_glm']

    # gpus = [2]
    # models_per_gpu = 4
    # algo_groups =  ['run_onestage_nonneural'] 
    
    print("============================algo groups: {} ==============================".format(algo_groups))
    timestr = time.strftime("%Y%m%d-%H%M")
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
                
    run_exps(args, algo_groups, result_path,gpus,models_per_gpu,timestr)