import argparse
import logging

def parse_args():
    def str2bool(v):
        if v.lower() in ['yes', 'true', 't', 'y','1']:
            return True
        elif v.lower() in ['no', 'false', 'f', 'n', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser() 

    # path
    parser.add_argument("--root_data_dir",type=str,default="/home/v-mezhang/blob/data/")
    parser.add_argument("--root_proj_dir",type=str,default="/home/v-mezhang/blob/CB4Rec/")
    # parser.add_argument("--root_proj_dir",type=str,default="./")
    parser.add_argument("--result_path", type=str, default='/home/v-mezhang/blob/CB4Rec/results/', help = 'CB simulation results')

    # Preprocessing 
    parser.add_argument("--dataset",type=str,default='large')
    parser.add_argument("--cb_train_ratio", type=float, default=0.2)
    parser.add_argument("--sim_npratio", type=int, default=4)
    parser.add_argument("--sim_val_batch_size", type=int, default=1024)
    parser.add_argument("--split_large_topic", type=str2bool, default=False)
    parser.add_argument("--pretrain_topic", type=str2bool, default=False)

    # Simulator
    parser.add_argument("--sim_path", type=str, default='./pretrained_models/sim_emp_ips_nrms_normalized_r14_ep5', help='nrms: simulator_pretrained_models/sim_nrms_bce_r14_ep6_thres038414; ips: ./pretrained_models/sim_emp_ips_nrms_normalized_r14_ep5') # "/home/v-mezhang/blob/model/large/large.pkl"
    parser.add_argument("--sim_threshold", type=float, default=0.38414)
    parser.add_argument("--ips_path", type=str, default="/home/thanhnt/projects/CB4Rec/runs/prop_pn=2-8_20220222_163423/model_best_4")
    parser.add_argument("--ips_normalize", type=bool, default=True)
    parser.add_argument("--empirical_ips", type=bool, default=True)
    parser.add_argument("--sim_margin", type=float, default=0.001)
    parser.add_argument("--reward_type", type=str, default='threshold_eps', help='soft/hard/hybrid/threshold/threshold_eps')

    # Simulation
    parser.add_argument("--algo",type=str,default="ts_neuralucb")
    parser.add_argument("--algo_prefix", type=str, default="algo",
        help='the name of save files')
    parser.add_argument("--n_trials", type=int, default=10, help = 'number of experiment runs')
    parser.add_argument("--num_selected_users", type=int, default=1000, help='number of randomly selected users from val set')
    parser.add_argument("--T", type=int, default=2000, help = 'number of rounds (interactions)')
    parser.add_argument("--topic_update_period", type=int, default=50, help = 'Update period for CB topic model')
    parser.add_argument("--update_period", type=int, default=100, help = 'Update period for CB item model')
    parser.add_argument("--n_inference", type=int, default=5, help='number of Monte Carlo samples of prediction. ')
    parser.add_argument("--rec_batch_size", type=int, default=5, help='recommendation size for each round.')
    parser.add_argument("--per_rec_score_budget", type=int, default=1000, help='budget for calculating scores, e.g. ucb, for each rec')
    parser.add_argument("--max_batch_size", type=int, default=256, help = 'Maximum batch size your GPU can fit in.')
    parser.add_argument("--pretrained_mode",type=str2bool,default=True, 
        help="Indicates whether to load a pretrained model. True: load from a pretrained model, False: no pretrained model ")
    parser.add_argument("--preinference_mode",type=str2bool,default=True, 
        help="Indicates whether to preinference news before each model update.")

    parser.add_argument("--uniform_init",type=str2bool,default=True, 
        help="For Thompson Sampling: Indicates whether to init ts parameters uniformly")
    parser.add_argument("--gamma", type=float, default=1.0, help='ucb parameter: mean + gamma * std.')

    parser.add_argument("--fix_user",type=str2bool,default=False, 
        help="Indicate whether to use fix set of users to run simulation. If true, then use trial 0 with given order.")
    parser.add_argument("--sim_sampleBern", type=str2bool,default=False,
        help="If True: sample from Bernoulli to get binary simulated reward; If False: use a threshold.")
    parser.add_argument("--eva_model_valid", type=str2bool,default=False,
        help="If True: evaluate model on validation dataset")
    parser.add_argument("--cb_pretrained_models",type=str,default="cb_pretrained_models_dim64")

    # for neural linear
    parser.add_argument("--lambda_prior", type=float, default=1.0, help = 'Prior for neural linear thompson sampling covariance diagonal term')
    # parser.add_argument("--latent_dim", type=int, default=400, help = 'latent representation dim for neural linear')
    parser.add_argument("--item_linear_update_period", type=int, default=1, help = 'Update period for CB item linear model (for neural linear model)')

    # nrms topic
    parser.add_argument("--dynamic_aggregate_topic", type=str2bool, default = False) # whether to dynamically aggregate small topic during simulation
    parser.add_argument("--min_item_size", type=int, default=1000)

    
    # nrms 
    parser.add_argument("--npratio", type=int, default=4) 
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--max_his_len", type=int, default=50)
    parser.add_argument("--min_word_cnt", type=int, default=1) # 5
    parser.add_argument("--max_title_len", type=int, default=30)
    parser.add_argument("--word_embedding_dim", type=int, default=300)
    parser.add_argument("--num_attention_heads", type=int, default=20)
    parser.add_argument("--attention_dim", type=int, default=20)
    parser.add_argument("--news_query_vector_dim", type=int, default=200)
    parser.add_argument("--news_dim", type=int,default=64)
    
    
    # model training
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=4)


    args = parser.parse_args()

    # logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
