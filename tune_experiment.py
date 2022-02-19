# Part of code from https://github.com/thanhnguyentang/offline_neural_bandits/blob/main/tune_realworld.py
import os
import subprocess
import time
import argparse
import random
import numpy as np


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


def create_commands(args, algo_group='ts_neuralucb'):
    commands = []
    if algo_group == 'tune_ts_neuralucb':
        for uniform_init in [True, False]:
            algo_prefix = algo_group  + 'TSUniInit' + str(uniform_init) 
            # + '-' + str(args.n_trials) + '-' + str(args.T) 
            log_path = os.path.join(args.root_proj_dir, 'logs', algo_prefix + '.log')
            commands.append("python run_experiment.py --algo {}  --algo_prefix {} --uniform_init {}> {}".format(algo_group, algo_prefix, uniform_init, log_path))
    else:
        algo_prefix = algo_group +'-perRecScoreBudget' + str(args.per_rec_score_budget) 
        # + '-' + str(args.n_trials) + '-' + str(args.T) 
        log_path = os.path.join(args.root_proj_dir, 'logs', algo_prefix + '.log')
        commands.append("python run_experiment.py --algo {} --algo_prefix {} > {}".format(algo_group, algo_prefix, log_path))
    return commands


def run_exps(args, algo_groups):
    commands = []
    for algo_group in algo_groups:
        commands += create_commands(args, algo_group)
    # random.shuffle(commands)
    multi_gpu_launcher(commands, [0,1,2,3], 1)

if __name__ == '__main__':
    from configs.mezhang_params import parse_args
    args = parse_args()

    algo_group = ['single_neuralucb', 'ts_neuralucb', 'greedy', 'single_linucb']
    # algo_group = ['tune_ts_neuralucb']
    run_exps(args, algo_group)