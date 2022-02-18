import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser() 

    # path
    parser.add_argument("--root_data_dir",type=str,default="/home/v-mezhang/blob/data/")
    parser.add_argument("--root_proj_dir",type=str,default="/home/v-mezhang/blob/CB4Rec/")
    # parser.add_argument("--root_proj_dir",type=str,default="./")
    parser.add_argument("--sim_path", type=str, default='pretrained_models/sim_nrms_bce_r14_ep6_thres038414') # "/home/v-mezhang/blob/model/large/large.pkl"
    parser.add_argument("--sim_threshold", type=float, default=0.38414)

    # Preprocessing 
    parser.add_argument("--dataset",type=str,default='large')
    parser.add_argument("--num_selected_users", type=int, default=1000, help='number of randomly selected users from val set')
    parser.add_argument("--cb_train_ratio", type=float, default=0.2)
    parser.add_argument("--sim_npratio", type=int, default=4)
    parser.add_argument("--sim_val_batch_size", type=int, default=1024)

    # Simulation
    parser.add_argument("--algo",type=str,default="ts_neuralucb")
    parser.add_argument("--algo_prefix", type=str, default="algo",
        help='the name of save files')
    parser.add_argument("--n_trials", type=int, default=4, help = 'number of experiment runs')
    parser.add_argument("--T", type=int, default=1000, help = 'number of rounds (interactions)')
    parser.add_argument("--update_period", type=int, default=100, help = 'Update period for CB model')
    parser.add_argument("--n_inference", type=int, default=5, help='number of Monte Carlo samples of prediction. ')
    parser.add_argument("--rec_batch_size", type=int, default=5, help='recommendation size for each round.')
    parser.add_argument("--per_rec_score_budget", type=int, default=1000, help='buget for calcuating scores, e.g. ucb, for each rec')
    parser.add_argument("--max_batch_size", type=int, default=256, help = 'Maximum batch size your GPU can fit in.')
    parser.add_argument("--pretrained_mode",type=bool,default=True, 
        help="Indicates whether to load a pretrained model. True: load from a pretrained model, False: no pretrained model ")
    parser.add_argument("--preinference_mode",type=bool,default=True, 
        help="Indicates whether to preinference news before each model update.")

    parser.add_argument("--uniform_init",type=bool,default=True, 
        help="For Thompson Sampling: Indicates whether to init ts parameters uniformly")
    parser.add_argument("--gamma", type=float, default=1.0, help='ucb parameter: mean + gamma * std.')

    
    # nrms 
    parser.add_argument("--npratio", type=int, default=4) 
    parser.add_argument("--max_his_len", type=int, default=50)
    parser.add_argument("--min_word_cnt", type=int, default=1) # 5
    parser.add_argument("--max_title_len", type=int, default=30)
    
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
