import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_data_dir",
        type=str,
        default=
        "/home/v-mezhang/blob/data/",
    )
    parser.add_argument("--model_path", type=str, default="/home/v-mezhang/blob/model/")
    parser.add_argument("--sim_path", type=str, default="/home/v-mezhang/blob/model/simulator.pkl")
    parser.add_argument("--out_path", type=str, default="/home/v-mezhang/blob/model/")

    parser.add_argument("--dataset",
                        type=str,
                        default='demo')
    parser.add_argument("--mode",
                        type=str,
                        default='cb')
    parser.add_argument("--sim_type",
                        type=str,
                        default='ips') # none, nrms, ips
    parser.add_argument("--dropout_flag",
                        type=bool,
                        default=True)
    parser.add_argument("--finetune_flag",
                        type=bool,
                        default=True)
    parser.add_argument("--fliter_user", # fliter CB simulation user from training data
                        type=bool,
                        default=True)
                    

    parser.add_argument("--npratio", type=int, default=4)
    parser.add_argument("--max_his_len", type=int, default=50)
    parser.add_argument("--min_word_cnt", type=int, default=1)
    parser.add_argument("--max_title_len", type=int, default=30)
    parser.add_argument("--eva_batch_size", type=int, default=1024)
    parser.add_argument("--n_inference", type=int, default=1)
    parser.add_argument("--n_exper", type=int, default=1)
    parser.add_argument("--policy", type=str, default='ucb')
    parser.add_argument("--policy_para", type=list, default=[0.1])
    parser.add_argument("--k", type=str, default='all')
    
    # model training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--data_format_str", type=str, default='%m/%d/%Y %I:%M:%S %p')
    parser.add_argument("--interval_time", type=int, default=3600)


    parser.add_argument("--num_words_title", type=int, default=24)
    parser.add_argument(
        "--word_embedding_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--news_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=20,
    )
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_steps_per_epoch", type=int, default=1000000)

    parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default=None,
        help="choose which ckpt to load and test"
    )
    args = parser.parse_args()

    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
