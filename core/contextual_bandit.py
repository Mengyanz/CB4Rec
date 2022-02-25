"""Define abstract CB algorithm. """

from operator import itemgetter
import numpy as np 
import pickle, os, json
from collections import defaultdict
from utils.data_util import load_cb_train_data, load_cb_valid_data
import os
import torch

class ContextualBanditLearner(object):
    def __init__(self, args, name='ContextualBanditLearner'):
        """Args:
                rec_batch_size: int, recommendation size. 
                n_inference: int, number of Monte Carlo samples of prediction. 
                pretrained_mode: bool, True: load from a pretrained model, False: no pretrained model 
        """
        self.name = name 
        print(name)
        self.args = args
        self.rec_batch_size = self.args.rec_batch_size
        self.per_rec_score_budget = self.args.per_rec_score_budget
        self.pretrained_mode = self.args.pretrained_mode 
        
        self.reset()

    def load_cb_learner(self, cb_learner_path = None, topic=False):
        """load pretrained 
        Args
            cb_learner_path: str, pretrained cb learner path
        """
        if self.pretrained_mode:
            if not os.path.exists(cb_learner_path):
                raise Exception("No cb learner pretrained for this trial!")
            try:
                if topic:
                    print('Load pre-trained CB topic learner on this trial from ', cb_learner_path)
                    self.topic_model.load_state_dict(torch.load(cb_learner_path))
                else:
                    print('Load pre-trained CB item learner on this trial from ', cb_learner_path)
                    self.model.load_state_dict(torch.load(cb_learner_path))
            except:
                print('Warning: Current algorithm has no model. Load checkpoint failed.')
        else:
            raise NotImplementedError()
        
    def set_clicked_history(self, init_clicked_history):
        """
        Args:
            init_click_history: list of init clicked history nindexs
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

    def sample_actions(self, uids):
        """Choose an action given a context. 
        
        Args:
            uids: a list of str uIDs (user ids). 

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
        pass 

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

    def reset(self):
        """Reset the CB learner to its initial state (do this for each experiment). """
        self.clicked_history = defaultdict(list) # a dict - key: uID, value: a list of str nIDs (clicked history) of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
        #TODO: reset the internal model here for each instance of `ContextualBanditLearner`


    def train(self):
        print('Note that `self.data_buffer` gets updated after every learner-simulator interaciton in `run_contextual_bandit`')
        print('size(data_buffer): {}'.format(len(self.data_buffer)))
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

    # contexts = contexts[:3]

    # num_exper = args.num_exper
    # num_round = args.num_round

    np.random.seed(2022)

    clicked_history_fn = os.path.join(args.root_data_dir, 'large/utils/train_clicked_history.pkl')
    with open(clicked_history_fn, 'rb') as fo: 
        train_clicked_history = pickle.load(fo)

    with open(os.path.join(args.root_data_dir, 'large/utils/cb_val_users.pkl'), 'rb') as fo: 
        cb_val_users = pickle.load(fo) 
        
    with open(os.path.join(args.root_data_dir, 'large/utils/subcategory_byorder.json'), 'r') as fo: 
        topic_list = json.load(fo)      

    h_items_all = [] 
    h_rewards_all = []

    algos_name = ''
    for a in algos:
        algos_name += (a.name+'-')
    result_path = os.path.join(args.root_proj_dir, 'results')
    if not os.path.exists(result_path):
        os.mkdir(result_path) 

    for e in range(args.n_trials):

        item_path = os.path.join(result_path, "items-{}-ninference{}-dynamic{}-{}-{}.npy".format(args.algo_prefix,str(args.n_inference),str(args.dynamic_aggregate_topic), e, args.T))
        reward_path = os.path.join(result_path, "rewards-{}-ninference{}-dynamic{}-{}-{}.npy".format(args.algo_prefix,str(args.n_inference),str(args.dynamic_aggregate_topic), e, args.T))
        if os.path.exists(reward_path):
            # if the trail reward is already stored, pass the trail. 
            print('{} exists.'.format(reward_path))
            continue

        # reset each CB learner
        [a.reset() for a in algos] 

        print('trial = {}'.format(e))
        # independents runs to show empirical regret means, std
      
        cb_learner_path = os.path.join(args.root_proj_dir, 'cb_pretrained_models', 'indices_{}.pkl'.format(e))
        if args.split_large_topic:
            cb_topic_learner_path = os.path.join(args.root_proj_dir, 'cb_topic_pretrained_models_large_topic_splited', 'indices_{}.pkl'.format(e))
        else:
            cb_topic_learner_path = os.path.join(args.root_proj_dir, 'cb_topic_pretrained_models', 'indices_{}.pkl'.format(e))
        print('Load pre-trained CB learner on this trial from ', cb_learner_path)
        [a.load_cb_learner(cb_learner_path) for a in algos]
        [a.load_cb_learner(cb_topic_learner_path, topic=True) for a in algos]

        if args.fix_user:
            load_idx = 0
        else:
            load_idx = e
        # Load the initial history for each user in each CB learner
        indices_path = os.path.join(args.root_proj_dir, 'meta_data', 'indices_{}.npy'.format(load_idx))
        # random_ids = np.load('./meta_data/indices_{}.npy'.format(e))
        random_ids = np.load(indices_path)
        user_set = [cb_val_users[j] for j in random_ids[:args.num_selected_users]]

        init_history = {key:train_clicked_history[key] for key in user_set}

        # Set the init clicked history in each CB learner 
        [a.set_clicked_history(init_history) for a in algos]

        
        # Run the contextual bandit process
        h_items = np.empty((len(algos), args.rec_batch_size,0), float)
        h_rewards = np.empty((len(algos), args.rec_batch_size,0), float) # (n_algos, rec_bs, T)
        for t in range(args.T):
            # iterate over selected users
            print('==========[trial = {}/{} | t = {}/{}]==============='.format(e, args.n_trials, t, args.T))
            
            if args.fix_user:
                u = user_set[t]
            else:
                # Randomly sample a user 
                u = np.random.choice(user_set) 
            print('user: {}.'.format(u))
            
            item_batches = []
            topic_batches = []
            for a in algos:
                topics, items = a.sample_actions(u) # recommend for user u using their current history 

                topic_batches.append(topics) 
                item_batches.append(items) 

            # action_batches = [a.sample_actions([context]).ravel() for a in algos] #(num_algos, args.rec_batch_size)
            print('  rec_topic: {}'.format(topic_batches))

            print('  rec_news: {}'.format(item_batches))


            reward_batches = [simulator.reward(u, items).ravel() for items in item_batches] #(num_algos, rec_batch_size)
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
                items = item_batches[j] 
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

            if t % 1000 == 0 and t > 0:
                temp_item_path = os.path.join(result_path, "items-{}-ninference{}-dynamic{}-{}-{}.npy".format(args.algo_prefix,str(args.n_inference),str(args.dynamic_aggregate_topic), e, args.T))
                temp_reward_path = os.path.join(result_path, "rewards-{}-ninference{}-dynamic{}-{}-{}.npy".format(args.algo_prefix,str(args.n_inference),str(args.dynamic_aggregate_topic), e, args.T))
                print('Debug h_items shape: ', np.expand_dims(h_items, axis=0).shape)
                print('Debug h_rewards shape: ', np.expand_dims(h_rewards, axis = 0).shape)
                np.save(temp_item_path, np.expand_dims(h_items, axis=0))
                np.save(temp_reward_path, np.expand_dims(h_rewards, axis = 0))

        h_items_all.append(h_items)
        h_rewards_all.append(h_rewards) # (n_trials, n_algos, rec_bs, T)
        print('Debug h_items shape: ', np.array(h_items_all).shape)
        print('Debug h_rewards shape: ', np.array(h_rewards_all).shape)
        np.save(item_path, np.array(h_items_all))
        np.save(reward_path, np.array(h_rewards_all))
    return np.array(h_items_all), np.array(h_rewards_all)

    


    

