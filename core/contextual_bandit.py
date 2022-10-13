"""Define abstract CB algorithm. """

from operator import itemgetter
import numpy as np 
import pickle, os, json
from collections import defaultdict
from utils.data_util import load_cb_train_data, load_cb_valid_data, load_word2vec, load_cb_topic_news
import os
import torch
from sklearn.metrics import roc_auc_score
import datetime
from tqdm import tqdm
from evaluate import cal_metric

class ContextualBanditLearner(object):
    def __init__(self, args, device, name='ContextualBanditLearner'):
        """Args:
                args: parameters loaded by config argparser 
                device: cuda device

                nid2index: dict, key: nid; value: nindex
                word2vec: dict, key: word index; value: word embedding (glove)
                nindex2vec: dict, key: nindex, value: list of news title word indexes
                
                cb_news: dict, key: topic string; value: list of news nindex under key topic
                cb_topics: list of topics
                news_topics: dict, key: news index, value: topic string

                rec_batch_size: int, recommendation size. 
                n_inference: int, number of Monte Carlo samples of prediction. 
                pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 
        """
        self.name = name 
        self.args = args
        self.device = device
        print('Debug self.device: ', self.device)

        self.nid2index, self.word2vec, self.nindex2vec = load_word2vec(args, 'utils')
        self.args.num_all_news = len(self.nid2index)
        print('num_all_news: ', self.args.num_all_news)
        
        topic_news = load_cb_topic_news(args) # dict, key: subvert; value: list nIDs 
        cb_news = defaultdict(list)
        news_topics = {} # key: news index, value: topic
        for k,v in topic_news.items():
            for l in v:
                nid = l.strip('\n').split("\t")[0] # get nIDs 
                cb_news[k].append(nid) 
                news_topics[self.nid2index[nid]]=k
        self.cb_news = cb_news 
        self.cb_topics = list(self.cb_news.keys())
        self.args.num_topics = len(self.cb_topics)
        print('num_topics: ', self.args.num_topics)
        
        if args.per_rec_score_budget >= int(1e8): # no comp budget
            self.min_item_size = int(self.args.num_all_news / self.args.num_topics)
            print('Set min_item_size to ', self.min_item_size)
        self.news_topics = news_topics
        self.topic_budget = 0 # the score budget for topic exploration
        # self.topic_budget = len(self.cb_topics) # for two stage

        self.rec_batch_size = self.args.rec_batch_size
        self.per_rec_score_budget = self.args.per_rec_score_budget
        self.pretrained_mode = self.args.pretrained_mode 
        # self.reset()

    def load_cb_learner(self, cb_learner_path = None, topic=False, cb_pretrain_data = None):
        """load pretrained models for topic or item
        Args
            cb_learner_path: str, pretrained cb learner path
        """
        if self.pretrained_mode:
            # if not os.path.exists(cb_learner_path):
            #     raise Exception("No cb learner pretrained for this trial!")
            if topic:
                try:
                    print('Load pre-trained CB topic learner on this trial from ', cb_learner_path)
                    self.topic_model.load_state_dict(torch.load(cb_learner_path))
                except:
                    print('Warning: Current algorithm has no topic model. Load checkpoint failed.')
            else:
                try: 
                    print('Load pre-trained CB item learner on this trial from ', cb_learner_path)
                    self.model.load_state_dict(torch.load(cb_learner_path))
                except:
                    print('Warning: Current algorithm has no item model. Load checkpoint failed.')
        else:
            print('In pretrained_mode False: use no pretrained model.')
            # raise NotImplementedError()
      
    def set_clicked_history(self, init_clicked_history):
        """
        Args:
            init_click_history: list of init clicked history nindexes
        """
        self.clicked_history = init_clicked_history

    def update_data_buffer(self, pos, neg, uid, t): 
        self.data_buffer.append([pos, neg, self.clicked_history[uid], uid, t]) 

    def update_clicked_history(self, pos, uid):
        """
        Args:
            pos: a list of str nIDs, positive news of uid 
            uid: str, user id 
        """
        for item in pos:
            if item not in self.clicked_history[uid]: #TODO: Is it OK for an item to appear multiple times in a clicked history?
                self.clicked_history[uid].append(item)

    # def create_cand_set(self, m=1):
    #     """sample candidate news set given score budget.
    #     Args:
    #         m: number of recommendations per iteration
    #     Return
    #         cand_news: sampled candidate news nindex
    #     """
    #     # TODO: to make full use of budget
    #     score_budget = self.per_rec_score_budget * m - int(self.topic_budget /(self.rec_batch_size/m))
    #     cand_news = [item for sublist in list(self.cb_news.values()) for item in sublist]
    #     cand_news = [self.nid2index[n] for n in cand_news]
    #     if len(cand_news)>score_budget:
    #         print('Randomly sample {} candidates news out of candidate news ({})'.format(score_budget, len(cand_news)))
    #         cand_news = np.random.choice(cand_news, size=score_budget, replace=False).tolist()
    #     return cand_news 

    def item_rec(self, uid, cand_news, m = 1): 
        """
        Args:
            uid: str, a user id 
            cand_news: a list of int (not nIDs but their index version from `nid2index`) 
            m: int, number of items to rec 

        Return: 
            items: a list of `len(uids)`int 
        """
        pass

    def sample_actions(self, uid, cand_news = None):
        """Choose an action given a context. 
        
        Args:
            uids: a str uIDs (user id). 
            cand_news: list of candidate news indexes 
        Return: 
            topics: (len(uids), `rec_batch_size`)
            items: (len(uids), `rec_batch_size`) 
        """
        # 

        ## Topic recommendation 
        # Recommend `len(uids)` topics using the current topic model. 

        ## Item recommendation 
        # * Compute the user representation of each `uids` user using the current clicked history `self.clicked_history` 
        # and the current `user_encoder``
        # * Compute the news representation using the current `news_encoder`` for each news of each recommended topics above 
        # * Run these two steps above for `n_inference` times to estimate score uncertainty 

        # cand_news = [item for sublist in list(self.cb_news.values()) for item in sublist]
        # cand_news = [self.nid2index[n] for n in cand_news]

        if cand_news is None:
            cand_news = list(range(self.args.num_all_news))
        
        rec_items = self.item_rec(uid, cand_news, self.rec_batch_size)

        return np.empty(0), rec_items

    def update(self, topics, items, rewards, mode = 'topic', uid = None):
        """Update its internal model. 

        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its integer index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`
            uid: user id.
        """
        # Update the topic model 

        # Update the user_encoder and news_encoder using `self.clicked_history`
        pass

    def reset(self, e=None, reload_flag=False, reload_path=None):
        """Save Reset the CB learner to its initial state (do this for each experiment).
        Args
            e: int
                trial
            reload_flag: bool
                indicates whether reload from saved file and continue to run each trial for more iterations
        """
        # TODO: currently reload is only added in neural greedy. 

        if reload_flag:
            model_path = os.path.join(self.args.result_path, self.args.dataset, 'model') # store final results
            with open(os.path.join(model_path, "{}_clicked_history.pkl".format(e)), 'rb') as f:
                self.clicked_history = pickle.load(f) 
            self.data_buffer = np.load(os.path.join(model_path, "{}_data_buffer".format(e)))
        else:
            self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
            self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  

    def train(self):
        print('Note that `self.data_buffer` gets updated after every learner-simulator interaciton in `run_contextual_bandit`')
        print('size(data_buffer): {}'.format(len(self.data_buffer)))
        pass

    def save(self, e=None):
        """Save the CB learner for future reload to continue run more iterations.
        Args
            e: int
                trial
        """
        try:
            model_path = os.path.join(self.args.result_path, self.args.dataset, 'model') # store final results
            with open(os.path.join(model_path, "{}_clicked_history.pkl".format(e)), "wb") as f:
                pickle.dump(self.clicked_history, f)
            np.save(os.path.join(model_path, "{}_data_buffer".format(e)), self.data_buffer)
            print('Info: model saved at {}'.format(model_path))
            print('Warning: only save clicked_history and data_buffer. Add more if needed.')
            # TODO: currently save is only added in neural greedy. 
        except AttributeError:
            print('Warning: no attribute clicked_history find. Skip saving.')
            pass 

def run_contextual_bandit(args, simulator, algos):
    """Run a contextual bandit problem on a set of algorithms.
    Args:
        contexts: A list of user samples. 
        simulator: An instance of Simulator.
        algos: List of algorithms (instances of `ContextualBanditLearner`) to use in the contextual bandit instance.
    Returns:
        h_actions: Matrix with actions: size (num_algorithms, rec_batch_size, num_context).
        h_rewards: Matrix with rewards: size (num_algorithms, rec_batch_size, num_context).
            where rec_batch_size (in args): int, number of recommendations per context.
    """

    np.random.seed(2022)
    clicked_history_fn = os.path.join(args.root_data_dir, args.dataset, 'utils/train_clicked_history.pkl')
    with open(clicked_history_fn, 'rb') as fo: 
        train_clicked_history = pickle.load(fo)

    with open(os.path.join(args.root_data_dir, args.dataset, 'utils/cb_val_users.pkl'), 'rb') as fo: 
        cb_val_users = pickle.load(fo) 
        
    with open(os.path.join(args.root_data_dir, args.dataset, 'utils/subcategory_byorder.json'), 'r') as fo: 
        topic_list = json.load(fo)      

    h_items_all = [] 
    h_rewards_all = []

    algo_names = []
    for a in algos:
        algo_names.append(a.name)

    # runs_path = os.path.join(args.result_path, 'runs') # store running results
    trial_path = os.path.join(args.result_path, 'trial') # store final results
    # if not os.path.exists(trial_path):
    #         os.mkdir(trial_path) 
    # all_path = os.path.join(args.result_path, 'all') # store all results

    t_start = datetime.datetime.now()
        
    for e in range(0, args.n_trials):      
        # store each trial results
        item_path = os.path.join(trial_path,  "{}-items-{}-{}.npy".format(e, args.algo_prefix, args.T))
        reward_path = os.path.join(trial_path, "{}-rewards-{}-{}.npy".format(e, args.algo_prefix, args.T))
        if os.path.exists(reward_path):
            # if the trail reward is already stored, pass the trail. 
            print('{} exists.'.format(reward_path))
            continue

        # reset each CB learner
        [a.reset(e, reload_flag=args.reload_flag, reload_path=args.reload_path) for a in algos] 

        print('trial = {}'.format(e))
        # independents runs to show empirical regret means, std
      
        cb_learner_path = os.path.join(args.root_proj_dir, args.cb_pretrained_models, args.dataset, 'indices_{}.pkl'.format(e))
        if args.split_large_topic:
            cb_topic_learner_path = os.path.join(args.root_proj_dir,'cb_topic_pretrained_models_large_topic_splited', args.dataset, 'indices_{}.pkl'.format(e))
        else:
            cb_topic_learner_path = os.path.join(args.root_proj_dir, 'cb_topic_pretrained_models', args.dataset, 'indices_{}.pkl'.format(e))
        # print('Load pre-trained CB learner on this trial from ', cb_learner_path)
        [a.load_cb_learner(cb_learner_path) for a in algos]
        [a.load_cb_learner(cb_topic_learner_path, topic=True) for a in algos]

        # if args.fix_user:
        #     load_idx = 0
        # else:
        #     load_idx = e

        # Load the initial history for each user in each CB learner
        indices_path = os.path.join(args.root_proj_dir, 'meta_data', args.dataset, 'indices_{}.npy'.format(e))
        # random_ids = np.load('./meta_data/indices_{}.npy'.format(e))
        random_ids = np.load(indices_path)
        user_set = [cb_val_users[j] for j in random_ids[:args.num_selected_users]]

        init_history = {key:train_clicked_history[key] for key in user_set}

        # Set the init clicked history in each CB learner 
        [a.set_clicked_history(init_history) for a in algos]

        if args.eva_model_valid:
            [a.run_eva() for a in algos]

        np.random.seed(2022)
        us = np.random.choice(user_set, size = args.T, replace=True)

        np.random.seed(2022) # REVIEW: keep sampled news the same from different trials, you can alternatively set to different seed to make news different for different trials.
        full_news_indexes = list(range(args.num_all_news))
        score_budget = min(args.per_rec_score_budget * args.rec_batch_size, len(full_news_indexes))
        newss = np.zeros((args.T, score_budget))
        for row in range(args.T):
            newss[row,:] = np.random.choice(full_news_indexes, size=score_budget, replace=False) # in each iteration, the candidate news is not repeatable
        # newss = np.random.choice(full_news_indexes, size=args.T * score_budget, replace=True).reshape(args.T, -1)
        newss = np.array(newss, dtype = int)

        # Run the contextual bandit process
        if args.reload_flag and args.reload_path is not None:
            root_path, reload_algo_prefix = args.reload_path.split('/model/')
            reload_item_path = os.path.join(root_path, 'trial', "{}-items-{}.npy".format(e, reload_algo_prefix))
            reload_reward_path = os.path.join(root_path, 'trial', "{}-rewards-{}.npy".format(e, reload_algo_prefix))
            h_items = np.array(np.load(reload_item_path))
            h_rewards = np.array(np.load(reload_reward_path))
            reload_t = h_rewards.shape[-1]
        else:
            h_items = np.empty((len(algos), args.rec_batch_size,0), float)
            h_rewards = np.empty((len(algos), args.rec_batch_size,0), float) # (n_algos, rec_bs, T)
            reload_t = 0

        rec_time = []
        for t in tqdm(range(reload_t, args.T)):
            # iterate over selected users
            print('==========[trial = {}/{} | t = {}/{}]==============='.format(e, args.n_trials, t, args.T))
            
            # if args.fix_user:
            #     u = user_set[t % args.num_selected_users]
            # else:
            #     # Randomly sample a user 
            #     u = np.random.choice(user_set) 
            u = us[t]
            print('user: {}.'.format(u))

            # print('Randomly sample {} candidates news out of candidate news ({})'.format(score_budget, len(full_news_indexes)))
            # cand_news_indexes = np.random.choice(full_news_indexes, size=score_budget, replace=False).tolist()
            cand_news_indexes = newss[t].tolist()
            # print('Debug cand news indexes: ', cand_news_indexes)

            
            item_batches = []
            item_batches_phcb = []
            topic_batches = []
            for a in algos:
                # specify cb pretrained path for glm model pretrain
                # a.cb_pretrained_path = os.path.join(args.root_data_dir, args.dataset, 'utils', "cb_train_contexts_nuser={}_splitratio={}_trial={}.pkl".format(args.num_selected_users, args.cb_train_ratio, e))
                time1 = datetime.datetime.now()
                if a.name.startswith('2_'): # two stage
                    topics, items = a.sample_actions(u) # recommend for user u using their current history 
                else:
                    print('For algorithm {}, use sampled candidate set ({}).'.format(a.name, len(cand_news_indexes)))
                    topics, items = a.sample_actions(u, cand_news_indexes) # recommend for user u using their current history 
                time2 = datetime.datetime.now()
                print('TIME: algorithm {} recommend topic+item used {}'.format(a.name, time2-time1))
                rec_time.append((time2-time1).total_seconds())

                topic_batches.append(topics) 
                if a.name.lower() != 'phcb':
                    item_batches.append(items)
                else:
                    item_batches.append(items[0])
                    item_batches_phcb.append(items)

            # action_batches = [a.sample_actions([context]).ravel() for a in algos] #(num_algos, args.rec_batch_size)
            print('  rec_topic: {}'.format(topic_batches))

            print('  rec_news: {}'.format(item_batches)) # print('  rec_news: {}'.format([item.gid for item in item_batches[0]]))

            # other algorithms, hcb/phcb
            reward_batches = [simulator.reward(u, items).ravel() if type(items[0]) is int else simulator.reward(u, [item.gid for item in items]).ravel()  for items in item_batches] #(num_algos, rec_batch_size)
            #@TODO: simulator has a complete history of each user, and it uses that complete history to simulate reward. 
            print('  rewards: {}'.format(reward_batches))

            # main
            # reward_batches = [simulator.reward([context], action_batch).ravel() for action_batch in action_batches] #(num_algos, args.rec_batch_size)
            # print('Rewards: {}'.format(reward_batches))
            # for j, a in enumerate(algos):
            #    a.update(context, topic_batches[j], action_batches[j], reward_batches[j])


            h_items = np.concatenate((h_items, np.array(item_batches)[:,:,None]), axis=2)
            h_rewards = np.concatenate((h_rewards, np.array(reward_batches)[:,:,None]), axis=2)


            # Update the data buffer and clicked history and models
            for j,a in enumerate(algos):
                topics = topic_batches[j]
                if a.name.lower() != 'phcb':
                    items = item_batches[j]
                else:
                    items = item_batches_phcb[j]
                rewards = reward_batches[j]
                pos = []
                neg = []
                for it,r in zip(items, rewards):
                    if r == 1:
                        pos.append(it) 
                    else:
                        neg.append(it)
                # WARNING: It's important to `update_data_buffer` before `update_clicked_history`
                #   because `update_data_buffer` use the current `clicked_history` 
                #@TODO: `pos` can contain multiple news. Consider further split `pos` further for `update_data_buffer` if necessary
                a.update_data_buffer(pos, neg, u, t) 
                a.update_clicked_history(pos, u)

                # Update the topic model 
                if t % args.topic_update_period == 0 and t > 0:
                # Topic model update each time period
                    a.update(topics, items, rewards, mode = 'topic') 

                if t % args.update_period == 0 and t > 0: # Update the item model (i.e. news_encoder and user_encoder)
                    a.update(topics, items, rewards, mode = 'item', uid = u)
                    
                if t % args.item_linear_update_period == 0 and t > 0: # Update the item model (i.e. news_encoder and user_encoder)
                    # for neural linear model only
                    a.update(topics, items, rewards, mode = 'item-linear', uid = u)

                # DEBUG for neural glmucb - lr model auc
                # if a.name == '2_neuralglmadducb':
                #     if (t + 1) % 10 == 0:
                #         targets = simulator.reward(u, cand_news_indexes).ravel()
                #         preds = a.predict(u, cand_news_indexes)
                #         auc = roc_auc_score(targets, preds)
                #         print('Debug at round {}: auc {}'.format(t, auc))
                #         a.writer.add_scalars('{} auc'.format(u), {'auc': auc}, t)
                    
            # if t % 1000 == 0 and t > 0:
            #     temp_item_path = os.path.join(result_path, "items-{}-ninference{}-dynamic{}-splitlarge{}-{}-{}.npy".format(args.algo_prefix,str(args.n_inference),str(args.dynamic_aggregate_topic),str(args.split_large_topic), e, args.T))
            #     temp_reward_path = os.path.join(result_path, "rewards-{}-ninference{}-dynamic{}-splitlarge{}-{}-{}.npy".format(args.algo_prefix,str(args.n_inference),str(args.dynamic_aggregate_topic),str(args.split_large_topic), e, args.T))
            #     print('Debug h_items shape: ', np.expand_dims(h_items, axis=0).shape)
            #     print('Debug h_rewards shape: ', np.expand_dims(h_rewards, axis = 0).shape)
            #     np.save(temp_item_path, np.expand_dims(h_items, axis=0))
            #     np.save(temp_reward_path, np.expand_dims(h_rewards, axis = 0))

                
            # if t % 1000 == 0 and t > 0:
            #     temp_item_path = os.path.join(result_path, "items-{}-{}-{}.npy".format(args.algo_prefix, e, t))
            #     temp_reward_path = os.path.join(result_path, "rewards-{}-{}-{}.npy".format(args.algo_prefix, e, t))
            #     print('Debug h_items shape: ', np.expand_dims(h_items, axis=0).shape)
            #     print('Debug h_rewards shape: ', np.expand_dims(h_rewards, axis = 0).shape)
            #     np.save(temp_item_path, np.expand_dims(h_items, axis=0))
            #     np.save(temp_reward_path, np.expand_dims(h_rewards, axis = 0))

        print('REPORT TIME: for trial {} eva rec time {} std {}'.format(e, np.mean(rec_time), np.std(rec_time)))    
        if (t+1)%100==0 or t == args.T -1:
            cal_metric(np.expand_dims(np.array(h_rewards), axis = 0), algo_names, ['ctr', 'cumu_reward'])

        t_now = datetime.datetime.now()
        print('TIME: run up to trial {} used {}'.format(e, t_now-t_start))

        # save each CB learner
        # [a.save(e) for a in algos] 

        # if (t+1) % args.save_freq == 0 or t == args.T -1:
        np.save(item_path, np.array(h_items)) # (n_algos, rec_bs, T)
        np.save(reward_path, np.array(h_rewards))
        print('Info saving rewards and item results at t: ', t)
        print('Debug h_items shape: ', np.array(h_items).shape)
        print('Debug h_rewards shape: ', np.array(h_rewards).shape)

    #     h_items_all.append(h_items)
    #     h_rewards_all.append(h_rewards) # (n_trials, n_algos, rec_bs, T)
    #     print('Debug h_items_all shape: ', np.array(h_items_all).shape)
    #     print('Debug h_rewards_all shape: ', np.array(h_rewards_all).shape)
        
        
    # return np.array(h_items_all), np.array(h_rewards_all)

    


    

