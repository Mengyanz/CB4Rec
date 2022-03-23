import os
import subprocess
import time
import argparse
import random
import numpy as np
import time

def get_gpu_memory_map(max_gpu_mem=None):
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    if max_gpu_mem is None: 
        max_gpu_mem = 1
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    result = result.decode('utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) / max_gpu_mem for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def multi_gpu_launcher(commands, gpus, models_per_gpu):
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

def create_commands(): 
    commands = [] 
    topic_disabled_flags = ['False'] 
    pretrained_cb_flags = ['True']
    reward_types = ['hard', 'soft', 'hybrid', 'threshold', 'threshold_eps', 'bern']
    for reward_type in reward_types: 
        for topic_disabled in topic_disabled_flags:
            for pretrained_cb in pretrained_cb_flags:
                commands.append('python run_experiment.py  --reward_type {} --topic_update_disabled {} --pretrained_cb {}'.format(reward_type,topic_disabled,pretrained_cb))

    return commands

def run_exps():
    MEM_FRAC = 0.6 
    commands = create_commands()
    gpu_map = get_gpu_memory_map(max_gpu_mem=32510)
    free_gpus = [] 
    for gpu_id, mem in gpu_map.items():
        if mem <= MEM_FRAC: 
            free_gpus.append(gpu_id)

    print(free_gpus)
    # random.shuffle(commands)
    multi_gpu_launcher(commands, free_gpus, 1)


if __name__ == '__main__':
    run_exps()