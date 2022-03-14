# Part of code from https://github.com/thanhnguyentang/offline_neural_bandits/blob/main/tune_realworld.py
import os
import subprocess
import time
import argparse
import random
import numpy as np
import time

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
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs:
        if p is not None:
            p.wait()


def create_commands(args, algo_group, result_path):
    commands = []
    if algo_group == 'test':
        for algo in ['glmucb']:
            num_selected_users = 10
            # algo_prefix = algo + '-num_selected_users' + str(num_selected_users)
            for epochs in [1, 5, 10]:
                for lr in [0.1, 0.01]:
                    algo_prefix = algo + '-num_selected_users' + str(num_selected_users) + '-epochs' + str(epochs) + '-lr' + str(lr)
                    # + '-' + str(args.n_trials) + '-' + str(args.T) 
                    log_path = os.path.join(result_path, algo_prefix + '.log')
                    commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --epochs {} --lr {} > {}".format(algo, algo_prefix, result_path, num_selected_users, epochs, lr, log_path))
            # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {}".format(algo, algo_prefix, result_path, num_selected_users))
    elif algo_group == 'tune_neural_linear':
        for algo in ['neuralbilinucb_hybrid']: # 'neural_linearts', 'neuralglmucb_uihybrid'
            for gamma in [0, 0.1, 0.5, 1]:
                algo_prefix = algo + '-gamma' + str(gamma)
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --gamma {} > {}".format(algo, algo_prefix, result_path, gamma, log_path))
                # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {}".format(algo, algo_prefix, result_path))
    elif algo_group == 'tune_ts_neuralucb':
        for uniform_init in [True, False]:
            algo_prefix = algo_group  + '-TSUniInit' + str(uniform_init) 
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --uniform_init {}> {}".format(algo_group, algo_prefix, result_path, uniform_init, log_path))
    elif algo_group == 'tune_dynamicTopic':
        for algo in ['ts_neuralucb', 'neuralucb_neuralucb']:
            for dynamic_aggregate_topic in [True, False]:
                algo_prefix = algo + '-dynamicTopic' + str(dynamic_aggregate_topic) 
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --dynamic_aggregate_topic {}> {}".format(algo, algo_prefix, result_path, dynamic_aggregate_topic, log_path))
    elif algo_group == 'tune_fix_user':
        # test var of 10 repeat over fix user/random user 
        for fix_user in [True, False]:
            sim_sampleBern = True
            n_trials = 10
            algo_prefix = algo_group + '-ts_neuralucb'  + '-FixUser' + str(fix_user) + '-SimSampleBern' + str(sim_sampleBern) 
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --fix_user {} --sim_sampleBern {} --n_trials {}> {}".format('ts_neuralucb', algo_prefix, result_path,  fix_user, sim_sampleBern, n_trials, log_path))
    elif algo_group == 'tune_topic_update_period':
        for topic_update_period in [1,10]:
            algo = 'neuralucb_neuralucb'
            algo_prefix = algo_group + '-' + algo + '-topicUpdate' + str(topic_update_period) 
            print('Debug algo_prefix: ', algo_prefix)
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --topic_update_period {} > {}".format(algo, algo_prefix, result_path,  topic_update_period, log_path))
    elif algo_group == 'baselines':
        for algo in ['single_neuralucb']: # , 'greedy', 'single_linucb'
            algo_prefix = algo_group + '-' + algo 
            print('Debug algo_prefix: ', algo_prefix)
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} > {}".format(algo, algo_prefix, result_path,  log_path))

    else:
        algo_prefix = algo_group +'-perRecScoreBudget' + str(args.per_rec_score_budget) 
        # + '-' + str(args.n_trials) + '-' + str(args.T) 
        log_path = os.path.join(result_path, algo_prefix + '.log')
        commands.append("python run_experiment.py --algo {} --algo_prefix {} > {}".format(algo_group, algo_prefix, result_path,  log_path))
    return commands


def run_exps(args, algo_groups, result_path):
    commands = []
    for algo_group in algo_groups:
        commands += create_commands(args, algo_group, result_path)
    # random.shuffle(commands)
    multi_gpu_launcher(commands, [1,2,3,4,5,6,7], 1)

if __name__ == '__main__':
    from configs.mezhang_params import parse_args
    # from configs.zhenyu_params import parse_args
    args = parse_args()

    # algo_group = ['single_neuralucb', 'ts_neuralucb', 'greedy', 'neuralucb_neuralucb'] # 'single_linucb'
    # algo_group = ['tune_ts_neuralucb']
    algo_groups = ['tune_neural_linear'] # tune_dynamicTopic
    timestr = time.strftime("%Y%m%d-%H%M")
    for algo_group in algo_groups:
        result_path = os.path.join(args.root_proj_dir, 'results', algo_group, timestr)
        if not os.path.exists(result_path):
            os.makedirs(result_path) 
        trial_path = os.path.join(result_path, 'trial') # store final results
        if not os.path.exists(trial_path):
                os.mkdir(trial_path) 
    run_exps(args, algo_groups, result_path)