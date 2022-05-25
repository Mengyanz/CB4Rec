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
        
            if '2_neuralucb' in algo_name:
                algo_name = algo_name.replace('2_neuralucb', '2_neuralucb_neuralucb_')
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


# timestr = '20220316-0643'
# algo_group = 'test'
# num_selected_users = 10
# for algo in ['glmucb']:
#         for epochs in [1, 5]:
#             for lr in [0.1, 0.01, 0.001]:
#                 algo_prefixes.append(algo + '-num_selected_users' + str(num_selected_users) + '-epochs' + str(epochs) + '-lr' + str(lr))
# for algo in ['neural_dropoutucb']: # , 'greedy', 'linucb'
#         algo_prefixes.append(algo_group + '-' + algo + '-num_selected_users' + str(num_selected_users))

# timestr = '20220317-0327'
# algo_group = 'single_stage'
# num_selected_users = 10
# for algo in ['neural_dropoutucb', 'greedy', 'linucb']:
#     algo_prefixes.append(algo_group + '-' + algo )

# timestr = '20220318-1322' #'20220316-0642'
# algo_group = 'tune_neural_linear'
# for algo in ['neural_glmadducb', 'NeuralGBiLinUCB']:
#     for gamma in [0, 0.1]: #[0, 0.1, 0.5, 1]:
#         algo_prefixes.append(algo + '-gamma' + str(gamma) + '-num_selected_users' + str(num_selected_users))

# timestr = '20220319-0633' # '20220316-0643'
# algo_group = 'tune_topic_update_period'
# for algo in ['2_neuralucb', '2_ts_neuralucb']:
#         if algo == '2_ts_neuralucb':
#             updates = [1]
#         else:
#             updates = [10,50,100]
#         for topic_update_period in updates:
#             algo_prefixes.append(algo + '-topicUpdate' + str(topic_update_period) + '-num_selected_users' + str(num_selected_users))
#             # algo_prefixes.append(algo_group + '-' + algo + '-topicUpdate' + str(topic_update_period))

# timestr = '20220322-0834'
# algo_group = 'tune_pretrainedMode_rewardType'
# for pretrained_mode in [True, False]:
#     for reward_type in ['soft', 'hybrid', 'hard', 'threshold']:
#         algo = 'greedy'
#         algo_prefixes.append(algo + '-pretrained' + str(pretrained_mode) + '-reward' + str(reward_type))

# timestr = '20220323-0738'
# algo_group = 'tune_pretrainedMode_nuser'
# for pretrained_mode in [True, False]:
#     for num_selected_users in [10, 100, 1000]:
#         reward_type = 'threshold_eps'
#         algo = 'greedy'
#         algo_prefixes.append(algo + '-pretrained' + str(pretrained_mode) + '-num_selected_users' + str(num_selected_users))

# timestr = '20220325-0242'
# algo_group = 'test_dropout'
# for algo in ['greedy', 'neural_dropoutucb', 'uniform_random']:
#     algo_prefixes.append(algo)

# timestr = '20220325-0535'
# algo_group = 'test_lin_glm_neural_ucb'
# for algo in ['linucb', 'glmucb', 'neural_linucb', 'neural_glmucb']:
#     algo_prefixes.append(algo) 

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

# timestr = '20220428-1453'
# algo_group = 'test_lin_glm_neural_ucb'
# for algo in ['glmucb']:
#     algo_prefixes.append(algo) 

# timestr = '20220429-1242'
# algo_group = 'run_onestage_neural'
# for num_selected_users in [100, 1000]:
#     for algo in ['greedy', 'neural_dropoutucb', 'neural_linucb', 'neural_glmucb', 'neural_glmadducb', 'neural_gbilinucb']:
#         algo_prefixes.append(algo + '_nuser' + str(num_selected_users)) 

# timestr = '20220430-0000'
# algo_group = 'run_onestage_nonneural'
# for num_selected_users in [100, 1000]:
#     for algo in ['uniform_random', 'linucb', 'glmucb']:
#         algo_prefixes.append(algo + '_nuser' + str(num_selected_users))

# timestr = '20220501-0546'
    # algo_group = 'run_onestage_neural'
    # for num_selected_users in [10, 100, 1000]:
    #     for glm_lr in [0.0001, 0.01]:
    #         for algo in ['neural_glmadducb', 'neural_gbilinucb']: # 'greedy', 'neural_dropoutucb', 'neural_linucb', 'neural_glmucb', 
    #             algo_prefixes.append(algo + '_nuser' + str(num_selected_users) + '_glmlr' + str(glm_lr))
    
    # timestr = '20220502-0355'
    # algo_group = 'test_reload'
    # algo_prefixes.append('greedy_T400_reloadTrue')
    # algo_prefixes.append('greedy_T400_reloadFalse')
    # algo_prefixes.append('greedy_T200_reloadFalse')
    
    # timestr = '20220503-1402'
    # algo_group = 'debug_decrease_after_100'
    # # 100 is the firs nrms update, try not to update
    # algo = 'neural_glmadducb'
    # for update_period in [100, 3000]:
    #     algo_prefixes.append(algo + '_update' + str(update_period))

    # timestr = '20220505-0108'
    # algo_group = 'run_onestage_neural'
    # for num_selected_users in [10]:
    #     for glm_lr in [1e-5, 1e-3]:
    #         for algo in ['neural_gbilinucb']: # 'greedy', 'neural_dropoutucb', 'neural_linucb', 'neural_glmucb', 
    #             algo_prefixes.append(algo + '_nuser' + str(num_selected_users) + '_glmlr' + str(glm_lr))

    # # timestr = '20220503-1154'
    # # timestr = '20220504-0650'
    # # timestr = '20220505-0632'
    # # timestr = '20220505-0910'
    # timestr = '20220506-0459'
    # algo_group = 'test_twostage'
    # n_inference = 1
    # gamma = 0
    # # algo = '2_neuralucb' # 
    # algo = '2_neuralgreedy' 
    # algo_prefixes.append(algo + '_ninf' + str(n_inference) + '_gamma' + str(gamma)) 

    # timestr = '20220506-0244'
    # algo_group = 'test_twostage'
    # algo = '2_random'
    # algo_prefixes.append(algo)

    # timestr = '20220505-0122'
    # algo_group = 'tune_glm'
    # for algo in ['neural_glmucb']:
    #     for num_selected_users in [100, 1000]:
    #         for glm_lr in [1e-1,1e-2,1e-3,1e-4]:
    #             algo_prefixes.append(algo + '_glmlr' + str(glm_lr) + '_nuser' + str(num_selected_users))

    # timestr = '20220506-0451'
    # algo_group = 'tune_dropout'
    # algo = 'neural_dropoutucb'
    # for num_selected_users in [100]: #  100, 1000
    #     for gamma in [0.05, 0.2]:
    #         algo_prefixes.append(algo + '_nuser' + str(num_selected_users) + '_gamma' + str(gamma))
    
    # timestr = '20220506-0108'
    # algo_group = 'run_onestage_neural'
    # T = 5000
    # for num_selected_users in [10]: #  100, 1000
    #     # for glm_lr in [1e-3,1e-4]: # 0.0001, 0.01
    #     for algo in ['neural_gbilinucb', 'neural_glmadducb']: # 'greedy', 'neural_dropoutucb', 'neural_linucb', 'neural_glmucb', 'neural_glmadducb', 
    #         if algo == 'neural_gbilinucb':
    #             glm_lr = 1e-3
    #         if algo == 'neural_glmadducb':
    #             glm_lr = 0.01
    #         algo_prefix = algo + '_nuser' + str(num_selected_users) + '_glmlr' + str(glm_lr) + '_T' + str(T)
    #         algo_prefixes.append(algo_prefix)
    
    # timestr = '20220507-0159'
    # algo_group = 'test_reset_buffer'
    # algo = 'greedy'
    # for reset in [True, False]:
    #     algo_prefix = algo + '_resetbuffer' + str(reset)
    #     algo_prefixes.append(algo_prefix)

    # timestr = '20220506-0733'
    # algo_group = 'test_reload'
    # algo = 'greedy'
    # reload_flag = False
    # for T in [2000]:
    #     algo_prefix = algo + '_T' + str(T) + '_reload' + str(reload_flag)
    #     algo_prefixes.append(algo_prefix)

   