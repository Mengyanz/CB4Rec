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
                print("runnning ", cmd)
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs:
        if p is not None:
            p.wait()

def run_exps(args, algo_groups, result_path, gpus,models_per_gpu:
    commands = []
    for algo_group in algo_groups:
        commands += create_commands(args, algo_group, result_path)
    # random.shuffle(commands)
    multi_gpu_launcher(commands,gpus,models_per_gpu)


def create_commands(args, algo_group, result_path):
    commands = []
    num_selected_users = 10
    if algo_group == 'test_proposed':
        for algo in ['neural_gbilinucb']: # 'neural_glmadducb', 
            for gamma in [0, 0.1]: # 0, 0.1, 0.5, 1
                algo_prefix = algo + '-gamma' + str(gamma)
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --gamma {} > {}".format(algo, algo_prefix, result_path, num_selected_users, gamma, log_path))
    elif algo_group == 'test_dropout':
        for algo in ['greedy', 'neural_dropoutucb', 'uniform_random']:
            algo_prefix = algo 
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, num_selected_users, log_path))
    elif algo_group == 'test_lin_glm_neural_ucb':
        for algo in ['linucb', 'glmucb', 'neural_linucb', 'neural_glmucb']:
            algo_prefix = algo 
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, num_selected_users, log_path))
    elif algo_group == 'tune_pretrainedMode_rewardType':
        for pretrained_mode in [True, False]:
            for reward_type in ['threshold']: # 'soft', 'hybrid', 'hard', 
                algo = 'greedy'
                algo_prefix = algo + '-pretrained' + str(pretrained_mode) + '-reward' + str(reward_type) 
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --pretrained_mode {} --reward_type {} > {}".format(algo, algo_prefix, result_path, pretrained_mode, reward_type, log_path))
    elif algo_group == 'tune_pretrainedMode_nuser':
        for pretrained_mode in [True, False]:
            for num_selected_users in [10, 100, 1000]:
                reward_type = 'threshold_eps'
                algo = 'greedy'
                algo_prefix = algo + '-pretrained' + str(pretrained_mode) + '-num_selected_users' + str(num_selected_users)
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --num_selected_users {} --result_path {} --pretrained_mode {} --reward_type {} > {}".format(algo, algo_prefix, num_selected_users, result_path, pretrained_mode, reward_type, log_path))

    elif algo_group == 'test':
        for algo in ['glmucb']:       
            # algo_prefix = algo + '-num_selected_users' + str(num_selected_users)
            for epochs in [1, 5]:
                for lr in [0.1, 0.01, 0.001]:
                    algo_prefix = algo + '-num_selected_users' + str(num_selected_users) + '-epochs' + str(epochs) + '-lr' + str(lr)
                    # + '-' + str(args.n_trials) + '-' + str(args.T) 
                    log_path = os.path.join(result_path, algo_prefix + '.log')
                    commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --epochs {} --lr {} > {}".format(algo, algo_prefix, result_path, num_selected_users, epochs, lr, log_path))
            # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {}".format(algo, algo_prefix, result_path, num_selected_users))
        for algo in ['neural_dropoutucb']: # , 'greedy', 'linucb'
            algo_prefix =  algo + '-num_selected_users' + str(num_selected_users)
            print('Debug algo_prefix: ', algo_prefix)
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, num_selected_users, log_path))
    elif algo_group == 'tune_neural_linear':
        for algo in ['NeuralGBiLinUCB', 'neural_glmadducb']: # 'neural_linearts', 'neural_glmadducb'
            for gamma in [0, 0.1]:
                algo_prefix = algo  + '-gamma' + str(gamma) + '-num_selected_users' + str(num_selected_users)
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --gamma {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, gamma, num_selected_users, log_path))
                # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {}".format(algo, algo_prefix, result_path))
    elif algo_group == 'tune_2_ts_neuralucb':
        for uniform_init in [True, False]:
            algo_prefix = 'TSUniInit' + str(uniform_init) 
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --uniform_init {}> {}".format(algo_group, algo_prefix, result_path, uniform_init, log_path))
    elif algo_group == 'tune_dynamicTopic':
        for algo in ['2_ts_neuralucb', '2_neuralucb_neuralucb']:
            for dynamic_aggregate_topic in [True]: # , False
                algo_prefix = algo + '-dynamicTopic' + str(dynamic_aggregate_topic) 
                # + '-' + str(args.n_trials) + '-' + str(args.T) 
                log_path = os.path.join(result_path, algo_prefix + '.log')
                commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --dynamic_aggregate_topic {}> {}".format(algo, algo_prefix, result_path, dynamic_aggregate_topic, log_path))
    elif algo_group == 'tune_fix_user':
        # test var of 10 repeat over fix user/random user 
        for fix_user in [True, False]:
            sim_sampleBern = True
            n_trials = 10
            algo_prefix =  '2_ts_neuralucb'  + '-FixUser' + str(fix_user) + '-SimSampleBern' + str(sim_sampleBern) 
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --fix_user {} --sim_sampleBern {} --n_trials {}> {}".format('2_ts_neuralucb', algo_prefix, result_path,  fix_user, sim_sampleBern, n_trials, log_path))
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
    elif algo_group == 'single_stage':
        for algo in ['neural_dropoutucb', 'greedy', 'linucb', 'glmucb']: # , 'greedy', 'linucb'
            algo_prefix = algo 
            print('Debug algo_prefix: ', algo_prefix)
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(result_path, algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} > {}".format(algo, algo_prefix, result_path,  log_path))

    else:
        algo_prefix = algo_group +'-ninference' + str(args.n_inference) + '-dynamic' + str(args.dynamic_aggregate_topic)+'-splitlarge' + str(args.split_large_topic) +'-perRecScoreBudget' + str(args.per_rec_score_budget) 
        # + '-' + str(args.n_trials) + '-' + str(args.T) 
        log_path = os.path.join(result_path, algo_prefix + '.log')
        commands.append("python run_experiment.py --algo {} --algo_prefix {} --result_path {} > {}".format(algo_group, algo_prefix, result_path,  log_path))
    return commands

if __name__ == '__main__':
    from configs.mezhang_params import parse_args
    # from configs.zhenyu_params import parse_args
    args = parse_args()

    # settings
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    gpus = [0,1]
    models_per_gpu = 2
    algo_groups =  ['test_lin_glm_neural_ucb'] 
    
    print("============================algo groups: {} ==============================".format(algo_groups))
    timestr = time.strftime("%Y%m%d-%H%M")
    for algo_group in algo_groups:
        result_path = os.path.join(args.root_proj_dir, 'results', algo_group, timestr)
        if not os.path.exists(result_path):
            os.makedirs(result_path) 
        trial_path = os.path.join(result_path, 'trial') # store final results
        if not os.path.exists(trial_path):
                os.mkdir(trial_path) 
                
    run_exps(args, algo_groups, result_path,gpus,models_per_gpu)