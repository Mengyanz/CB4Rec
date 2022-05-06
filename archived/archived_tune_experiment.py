# elif algo_group == 'tune_fix_user':
#     # test var of 10 repeat over fix user/random user 
#     for fix_user in [True, False]:
#         sim_sampleBern = True
#         n_trials = 10
#         algo_prefix =  '2_ts_neuralucb'  + '-FixUser' + str(fix_user) + '-SimSampleBern' + str(sim_sampleBern) 
#         # + '-' + str(args.n_trials) + '-' + str(args.T) 
#         log_path = os.path.join(result_path, algo_prefix + '.log')
#         commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --fix_user {} --sim_sampleBern {} --n_trials {}> {}".format('2_ts_neuralucb', algo_prefix, result_path,  fix_user, sim_sampleBern, n_trials, log_path))
#         algo_prefixes.append(algo_prefix)
# elif algo_group == 'tune_2_ts_neuralucb':
#     for uniform_init in [True, False]:
#         algo_prefix = 'TSUniInit' + str(uniform_init) 
#         # + '-' + str(args.n_trials) + '-' + str(args.T) 
#         log_path = os.path.join(result_path, algo_prefix + '.log')
#         commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --uniform_init {}> {}".format(algo_group, algo_prefix, result_path, uniform_init, log_path))
#         algo_prefixes.append(algo_prefix)
# elif algo_group == 'test':
#     for algo in ['glmucb']:       
#         # algo_prefix = algo + '-num_selected_users' + str(num_selected_users)
#         for epochs in [1, 5]:
#             for lr in [0.1, 0.01, 0.001]:
#                 algo_prefix = algo + '-num_selected_users' + str(num_selected_users) + '-epochs' + str(epochs) + '-lr' + str(lr)
#                 # + '-' + str(args.n_trials) + '-' + str(args.T) 
#                 log_path = os.path.join(result_path, algo_prefix + '.log')
#                 commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --epochs {} --lr {} > {}".format(algo, algo_prefix, result_path, num_selected_users, epochs, lr, log_path))
#                 algo_prefixes.append(algo_prefix)
#         # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {}".format(algo, algo_prefix, result_path, num_selected_users))
#     for algo in ['neural_dropoutucb']: # , 'greedy', 'linucb'
#         algo_prefix =  algo + '-num_selected_users' + str(num_selected_users)
#         print('Debug algo_prefix: ', algo_prefix)
#         # + '-' + str(args.n_trials) + '-' + str(args.T) 
#         log_path = os.path.join(result_path, algo_prefix + '.log')
#         commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, num_selected_users, log_path))
#         algo_prefixes.append(algo_prefix)
# elif algo_group == 'tune_pretrainedMode_nuser':
#     for pretrained_mode in [True, False]:
#         for num_selected_users in [10, 100, 1000]:
#             reward_type = 'threshold_eps'
#             algo = 'greedy'
#             algo_prefix = algo + '-pretrained' + str(pretrained_mode) + '-num_selected_users' + str(num_selected_users)
#             # + '-' + str(args.n_trials) + '-' + str(args.T) 
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --num_selected_users {} --result_path {} --pretrained_mode {} --reward_type {} > {}".format(algo, algo_prefix, num_selected_users, result_path, pretrained_mode, reward_type, log_path))
#             algo_prefixes.append(algo_prefix)
# elif algo_group == 'tune_pretrainedMode_rewardType':
#     for pretrained_mode in [True, False]:
#         for reward_type in ['threshold']: # 'soft', 'hybrid', 'hard', 
#             algo = 'greedy'
#             algo_prefix = algo + '-pretrained' + str(pretrained_mode) + '-reward' + str(reward_type) 
#             # + '-' + str(args.n_trials) + '-' + str(args.T) 
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --pretrained_mode {} --reward_type {} > {}".format(algo, algo_prefix, result_path, pretrained_mode, reward_type, log_path))
#             algo_prefixes.append(algo_prefix)
# elif algo_group == 'test_lin_glm_neural_ucb':
#     for algo in ['glmucb']: # 'linucb', 'glmucb', 'neural_linucb', 'neural_glmucb'
#         algo_prefix = algo 
#         # + '-' + str(args.n_trials) + '-' + str(args.T) 
#         log_path = os.path.join(result_path, algo_prefix + '.log')
#         commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, num_selected_users, log_path))
#         algo_prefixes.append(algo_prefix)