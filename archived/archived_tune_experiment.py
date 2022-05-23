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

# 20220523
# elif algo_group == 'accelerate_bilinear':
#     for algo in ['2_neuralglmbilinucb']:
#         for dynamic_aggregate_topic in [True]:
#             algo_prefix = algo + '_dynTopic' + str(dynamic_aggregate_topic)
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --dynamic_aggregate_topic {} > {}".format(algo, algo_prefix, result_path, dynamic_aggregate_topic, log_path))
#             algo_prefixes.append(algo_prefix)
#  elif algo_group == 'test_reset_buffer':
#     algo = 'greedy'
#     # for reset in [True, False]:
#     #     algo_prefix = algo + '_resetbuffer' + str(reset)
#     #     log_path = os.path.join(result_path, algo_prefix + '.log')
#     #     commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --reset_buffer {}  > {}".format(algo, algo_prefix, result_path, reset, log_path))
#     #     algo_prefixes.append(algo_prefix)
#     update_period = int(1e8) # no update
#     algo_prefix = algo + '_update' + str(update_period)
#     log_path = os.path.join(result_path, algo_prefix + '.log')
#     commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --update_period {}  > {}".format(algo, algo_prefix, result_path, update_period, log_path))
#     algo_prefixes.append(algo_prefix)
# elif algo_group == 'tune_glm':
#     for algo in ['neural_glmucb']:
#         for num_selected_users in [100, 1000]:
#             for glm_lr in [1e-1,1e-2,1e-3,1e-4]:
#                 algo_prefix = algo + '_glmlr' + str(glm_lr) + '_nuser' + str(num_selected_users) #+ 'score_budget' + str(score_budget)
#                 # + '-' + str(args.n_trials) + '-' + str(args.T) 
#                 log_path = os.path.join(result_path, algo_prefix + '.log')
#                 commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --glm_lr {} > {}".format(algo, algo_prefix, result_path, num_selected_users, glm_lr, log_path))
#                 algo_prefixes.append(algo_prefix)
# elif algo_group == 'debug_decrease_after_100':
#         # 100 is the firs nrms update, try not to update
#         algo = 'neural_glmadducb'
#         n_trial = 10
#         for update_period in [100, 3000]:
#             algo_prefix = algo + '_update' + str(update_period)
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --update_period {} --n_trial {} > {}".format(algo, algo_prefix, result_path, update_period, n_trial, log_path))
#             algo_prefixes.append(algo_prefix)

# elif algo_group == 'test_reload':
#     algo = 'greedy'
#     n_trial = 5
#     # reward_type = 'threshold' # remove stochastic to check reload 

#     # reload_flag = True
#     # reload_path = os.path.join(args.root_proj_dir, 'results', 'test_reload', '20220502-0355', 'model', 'greedy_T200_reloadFalse-200')
#     reload_flag = False
#     reload_path = None

#     for T in [2000]:
#         algo_prefix = algo + '_T' + str(T) + '_reload' + str(reload_flag)
#         log_path = os.path.join(result_path, algo_prefix + '.log')
#         commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --T {} --reload_flag {} --reload_path {}  --n_trial {} > {}".format(algo, algo_prefix, result_path, T, reload_flag, reload_path, n_trial, log_path))
#         algo_prefixes.append(algo_prefix)
# elif algo_group == 'test_save_load':
#     T = 500
#     for algo in ['greedy']:
#         algo_prefix = algo 
#         log_path = os.path.join(result_path, algo_prefix + '.log')
#         commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --T {} > {}".format(algo, algo_prefix, result_path, T, log_path))
#         algo_prefixes.append(algo_prefix)
# elif algo_group == 'run_onestage_nonneural':
#     for num_selected_users in [100, 1000]:
#         for algo in ['uniform_random', 'linucb', 'glmucb']:
#             algo_prefix = algo + '_nuser' + str(num_selected_users) 
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, num_selected_users, log_path))
#             algo_prefixes.append(algo_prefix)
#  elif algo_group == 'tune_dropout':
#         for algo in ['greedy','neural_dropoutucb']:
#             for num_selected_users in [10, 100, 1000]: #  100, 1000
#             # for gamma in [0.01, 0.05, 0.5, 1]:
#                 gamma = 0.1
#                 algo_prefix = algo + '_nuser' + str(num_selected_users) + '_gamma' + str(gamma)
#                 log_path = os.path.join(result_path, algo_prefix + '.log')
#                 commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --gamma {} > {}".format(algo, algo_prefix, result_path, num_selected_users, gamma, log_path))
#                 algo_prefixes.append(algo_prefix)
# elif algo_group == 'debug_glm':
#     # T = 100
#     # score_budget = 20
#     # for num_selected_users in [10, 100, 1000]:
#     #     for algo in ['neural_glmucb']: # ['neural_glmucb_lbfgs']: # ['neural_glmucb', 'neural_linucb']: 
#     #         for glm_lr in [0.1, 0.001, 0.0001]:
#     #             algo_prefix = algo + '_glmlr' + str(glm_lr) #+ 'score_budget' + str(score_budget)
#     #             # + '-' + str(args.n_trials) + '-' + str(args.T) 
#     #             log_path = os.path.join(result_path, algo_prefix + '.log')
#     #             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --glm_lr {} > {}".format(algo, algo_prefix, result_path, num_selected_users, glm_lr, log_path))
#     #             algo_prefixes.append(algo_prefix)
#     algo = 'neural_dropoutucb'
#     for num_selected_users in [100]: #  100, 1000
#         for gamma in [0.05, 0.5, 1, 5]:
#             algo_prefix = algo + '_nuser' + str(num_selected_users) + '_gamma' + str(gamma)
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --gamma {} > {}".format(algo, algo_prefix, result_path, num_selected_users, gamma, log_path))
#             algo_prefixes.append(algo_prefix)
# elif algo_group == 'test_proposed':
#     for algo in ['neural_gbilinucb', 'neural_glmadducb']: # 'neural_glmadducb', 
#         for gamma in [0.1]: # 0, 0.1, 0.5, 1
#             algo_prefix = algo + '-gamma' + str(gamma)
#             # + '-' + str(args.n_trials) + '-' + str(args.T) 
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --gamma {} > {}".format(algo, algo_prefix, result_path, num_selected_users, gamma, log_path))
#             algo_prefixes.append(algo_prefix)
# elif algo_group == 'test_dropout':
#     for algo in ['greedy', 'neural_dropoutucb', 'uniform_random']:
#         algo_prefix = algo 
#         # + '-' + str(args.n_trials) + '-' + str(args.T) 
#         log_path = os.path.join(result_path, algo_prefix + '.log')
#         commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, num_selected_users, log_path))
#         algo_prefixes.append(algo_prefix)

# elif algo_group == 'tune_neural_linear':
#     for algo in ['NeuralGBiLinUCB', 'neural_glmadducb']: # 'neural_linearts', 'neural_glmadducb'
#         for gamma in [0, 0.1]:
#             algo_prefix = algo  + '-gamma' + str(gamma) + '-num_selected_users' + str(num_selected_users)
#             # + '-' + str(args.n_trials) + '-' + str(args.T) 
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --gamma {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path, gamma, num_selected_users, log_path))
#             # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {}".format(algo, algo_prefix, result_path))
#             algo_prefixes.append(algo_prefix)
# elif algo_group == 'tune_dynamicTopic':
#     for algo in ['2_ts_neuralucb', '2_neuralucb']:
#         for dynamic_aggregate_topic in [True]: # , False
#             algo_prefix = algo + '-dynamicTopic' + str(dynamic_aggregate_topic) 
#             # + '-' + str(args.n_trials) + '-' + str(args.T) 
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --dynamic_aggregate_topic {}> {}".format(algo, algo_prefix, result_path, dynamic_aggregate_topic, log_path))
#             algo_prefixes.append(algo_prefix)

# elif algo_group == 'tune_topic_update_period':
#     for algo in ['2_neuralucb', '2_ts_neuralucb']:
#         if algo == '2_ts_neuralucb':
#             updates = [1]
#         else:
#             updates = [10,50,100]
#         for topic_update_period in updates:
#             algo_prefix = algo + '-topicUpdate' + str(topic_update_period) + '-num_selected_users' + str(num_selected_users)
#             print('Debug algo_prefix: ', algo_prefix)
#             # + '-' + str(args.n_trials) + '-' + str(args.T) 
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --topic_update_period {} --num_selected_users {} > {}".format(algo, algo_prefix, result_path,  topic_update_period, num_selected_users, log_path))
#             algo_prefixes.append(algo_prefix)
# elif algo_group == 'single_stage':
#     for algo in ['neural_dropoutucb', 'greedy', 'linucb', 'glmucb']: # , 'greedy', 'linucb'
#         algo_prefix = algo 
#         print('Debug algo_prefix: ', algo_prefix)
#         # + '-' + str(args.n_trials) + '-' + str(args.T) 
#         log_path = os.path.join(result_path, algo_prefix + '.log')
#         commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} > {}".format(algo, algo_prefix, result_path,  log_path))
#         algo_prefixes.append(algo_prefix)
# elif algo_group == 'run_onestage_neural':
#         for num_selected_users in [10, 100, 1000]: 
#             # for glm_lr in [1e-3,1e-4]: # 0.0001, 0.01
#             for algo in ['greedy', 'neural_dropoutucb', 'neural_linucb', 'neural_glmucb', 'neural_glmadducb', 'neural_gbilinucb']: 
#                 if algo == 'neural_gbilinucb':
#                     glm_lr = 1e-3
#                 else:
#                     glm_lr = args.glm_lr
#                 algo_prefix = algo + '_nuser' + str(num_selected_users) + '_glmlr' + str(glm_lr) + '_T' + str(T)
#                 log_path = os.path.join(result_path, algo_prefix + '.log')
#                 commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --glm_lr {} > {}".format(algo, algo_prefix, result_path, num_selected_users, glm_lr, log_path))
#                 algo_prefixes.append(algo_prefix)
#         # T = 10000
#         # for num_selected_users in [10]: #  10, 100, 1000
#         #     # for glm_lr in [1e-3,1e-4]: # 0.0001, 0.01
#         #     for algo in ['neural_glmadducb', 'neural_gbilinucb']: # 'greedy', 'neural_dropoutucb', 'neural_linucb', 'neural_glmucb', 'neural_glmadducb', 'neural_gbilinucb', 
#         #         if algo == 'neural_gbilinucb':
#         #             glm_lr = 1e-3
#         #         else:
#         #             glm_lr = args.glm_lr
#         #         algo_prefix = algo + '_nuser' + str(num_selected_users) + '_glmlr' + str(glm_lr) + '_T' + str(T)
#         #         log_path = os.path.join(result_path, algo_prefix + '.log')
#         #         commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --glm_lr {} > {}".format(algo, algo_prefix, result_path, num_selected_users, glm_lr, log_path))
#         #         algo_prefixes.append(algo_prefix)
# if algo_group == 'test_largeT':
#         T = 10000
#         for algo in ['neural_glmadducb', 'neural_gbilinucb', '2_neuralglmadducb', '2_neuralglmbilinucb']:
            # if 'bilinucb' in algo:
            #     glm_lr = 1e-3
            # else:
            #     glm_lr = args.glm_lr
#             algo_prefix = algo + '_glmlr' + str(glm_lr) + '_T' + str(T)
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {}  --T {} > {}".format(algo, algo_prefix, result_path, T, log_path))
#             algo_prefixes.append(algo_prefix)
#     elif algo_group == 'tune_gamma':
#         for gamma in [0.01, 0.05, 0.1, 0.5, 1, 2]:
#             for algo in ['2_neuralglmadducb', '2_neuralglmbilinucb']:
#                 algo_prefix = algo + '_gamma' + str(gamma)
#                 log_path = os.path.join(result_path, algo_prefix + '.log')
#                 commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {}  --gamma {} > {}".format(algo, algo_prefix, result_path, gamma, log_path))
#                 algo_prefixes.append(algo_prefix)

#     elif algo_group == 'test_dynamic_topic':
#         for algo in ['2_neuralglmbilinucb', '2_neuralglmbilinucb']:
#             for dynamic_aggregate_topic in [True, False]:
#                 algo_prefix = algo + '_dynTopic' + str(dynamic_aggregate_topic) #+ '_glmlr' + str(glm_lr)
#                 log_path = os.path.join(result_path, algo_prefix + '.log')
#                 commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --dynamic_aggregate_topic {} > {}".format(algo, algo_prefix, result_path, dynamic_aggregate_topic, log_path))
#                 algo_prefixes.append(algo_prefix)

#     elif algo_group == 'test_rec_size':
        
#         # for algo in ['2_neuralgreedy', '2_neuralucb', '2_neuralglmadducb', '2_neuralglmbilinucb']: 
#         for algo in ['2_neuralglmadducb', '2_neuralglmbilinucb']: 
#          # 'uniform_random', 'greedy', 'neural_dropoutucb',  'neural_glmadducb', 'neural_gbilinucb', '2_random','2_neuralgreedy', '2_neuralucb', '2_neuralglmadducb', '2_neuralglmbilinucb'
#             for rec_batch_size in [1]: # 1， 10
#                 if rec_batch_size == 1:
#                     if 'adducb' or 'bilinucb' in algo:
#                         item_linear_update_period = 1
#                 else:
#                     item_linear_update_period = args.item_linear_update_period
#                 per_rec_score_budget = int(5000/rec_batch_size)
#                 min_item_size = per_rec_score_budget
#                 algo_prefix = algo + '_recSize' + str(rec_batch_size) + '_minPerTopic' + str(min_item_size) + '_glmUpdate' + str(item_linear_update_period)
#                 log_path = os.path.join(result_path, algo_prefix + '.log')
#                 commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --rec_batch_size {} --per_rec_score_budget {} --min_item_size {} --item_linear_update_period {}  > {}".format(algo, algo_prefix, result_path, rec_batch_size, per_rec_score_budget,  min_item_size, item_linear_update_period, log_path))
#                 algo_prefixes.append(algo_prefix)

#     elif algo_group == 'test_twostage':
#         # for algo in ['2_neuralucb']: # ['2_neuralgreedy', '2_neuralucb']: # 
#         #     for gamma in [0, 0.1, 0.5, 1]: # 
#         #         algo_prefix = algo + '_gamma' + str(gamma) 
#         #         log_path = os.path.join(result_path, algo_prefix + '.log')
#         #         # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} > {}".format(algo, algo_prefix, result_path, log_path))
#         #         commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --gamma {} > {}".format(algo, algo_prefix, result_path, gamma, log_path))
#         #         algo_prefixes.append(algo_prefix)
        
#         for algo in ['2_neuralglmadducb']: # ['2_neuralucb']: # , '2_neuralglmadducb', '2_neuralglmbilinucb'
#             reset_buffer = True
#             algo_prefix = algo + '_resetBuffer' + str(reset_buffer)
            
#             log_path = os.path.join(result_path, algo_prefix + '.log')
#             commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --reset_buffer {}  > {}".format(algo, algo_prefix, result_path, reset_buffer, log_path))
#             # commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --reset_buffer {} ".format(algo, algo_prefix, result_path, reset_buffer))
#             algo_prefixes.append(algo_prefix)

#     elif algo_group == 'run_onestage_neural':
#         for num_selected_users in [10, 100, 1000]: 
#             # for glm_lr in [1e-3,1e-4]: # 0.0001, 0.01
#             for algo in ['greedy', 'neural_dropoutucb', 'neural_linucb', 'neural_glmucb', 'neural_glmadducb', 'neural_gbilinucb']: 
#                 algo_prefix = algo + '_nuser' + str(num_selected_users) + '_glmlr' + str(glm_lr) + '_T' + str(T)
#                 log_path = os.path.join(result_path, algo_prefix + '.log')
#                 commands.append("python run_experiment.py --algo {}  --algo_prefix {} --result_path {} --num_selected_users {} --glm_lr {} > {}".format(algo, algo_prefix, result_path, num_selected_users, glm_lr, log_path))
#                 algo_prefixes.append(algo_prefix)

# algo_groups = ['test_reset_buffer']
# algo_groups = ['test_twostage']
# algo_groups = ['test_rec_size']
# algo_groups = ['tune_dropout']
# algo_groups = ['debug_glm']
# algo_groups =  ['run_onestage_neural'] 
# algo_groups = ['tune_glm']
# algo_groups = ['test_dynamic_topic']
# algo_groups = ['accelerate_bilinear']
# algo_groups = ['tune_gamma']

# timestr = '20220515-1445'
# timestr = '20220516-1100'
# timestr = '20220516-0859'
# timestr = '20220517-0125'
# timestr = '20220517-0310'
# timestr = '20220517-0855'
# timestr = '20220517-1610'
# timestr = '20220518-0112'