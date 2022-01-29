"""Define abstract CB algorithm. """

from operator import itemgetter
import numpy as np 
import pickle, os
from collections import defaultdict
from utils.data_util import load_cb_train_data, load_cb_valid_data

class ContextualBanditLearner(object):
    def __init__(self, args, rec_batch_size = 1, name='ContextualBanditLearner'):
        self.name = name 
        print(name)
        self.args = args
        self.rec_batch_size = rec_batch_size
        
        self.reset()
        
    def set_clicked_history(self, init_clicked_history):
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
            if item not in self.clicked_history[uid]:
                self.clicked_history[uid].append(item)

    def sample_actions(self, uids):
        """Choose an action given a context. 
        
        Args:
            uids: a list of str uIDs (user ids). 

        Return: 
            topics: (len(uids), `rec_batch_size`)
            items: (len(uids), `rec_batch_size`) @TODO: what if one topic has less than `rec_batch_size` numbers of items? 
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

    def update(self, topics, items, rewards, mode = 'topic'):
        """Update its internal model. 

        Args:
            topics: list of `rec_batch_size` str
            items: a list of `rec_batch_size` item index (not nIDs, but its integer index from `nid2index`) 
            rewards: a list of `rec_batch_size` {0,1}
            mode: `topic`/`item`

        @TODO: they recommend `rec_batch_size` topics 
            and each of the topics they recommend an item (`rec_batch_size` items in total). 
            What if one item appears more than once in the list of `rec_batch_size` items? 
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
        print('TODO: Train `self.model` using `self.data_buffer.')
        print('Note that `self.data_buffer` gets updated after every learner-simulator interaciton in `run_contextual_bandit`')
        print('size(data_buffer): {}'.format(len(self.data_buffer)))
        pass


def run_contextual_bandit(args, simulator, rec_batch_size, algos):
    """Run a contextual bandit problem on a set of algorithms.
    Args:
        contexts: A list of user samples. 
        simulator: An instance of Simulator.
        rec_batch_size: int, number of recommendations per context.
        algos: List of algorithms (instances of `ContextualBanditLearner`) to use in the contextual bandit instance.
    Returns:
        h_actions: Matrix with actions: size (num_algorithms, num_context, batch_size).
        h_rewards: Matrix with rewards: size (num_algorithms, num_context).
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

    for e in range(args.n_trials):
        h_items_all = [] 
        h_rewards_all = []

        # reset each CB learner
        [a.reset() for a in algos] 

        print('trial = {}'.format(e))
        # independents runs to show empirical regret means, std

        # @TODO: Load the CB train data for this trial and pre-train each CB learner on this
        print('TODO: Load the CB train data for this trial and pre-train each CB learner on this trial.')

        # Load the initial history for each user in each CB learner
        random_ids = np.load('./meta_data/indices_{}.npy'.format(e))
        user_set = [cb_val_users[j] for j in random_ids[:args.num_selected_users]]

        init_history = {key:train_clicked_history[key] for key in user_set}

        # Set the init clicked history in each CB learner 
        [a.set_clicked_history(init_history) for a in algos]

        
        # Run the contextual bandit process
        h_items = np.empty((len(algos), rec_batch_size,0), float)
        h_rewards = np.empty((len(algos), rec_batch_size,0), float) # (n_algos, rec_bs, T)
        for t in range(args.T):
            # iterate over selected users
            print('==========[trial = {}/{} | t = {}/{}]==============='.format(e, args.n_trials, t, args.T))
            
            # Randomly sample a user 
            u = np.random.choice(user_set) 
            print('user: {}.'.format(u))
            
            item_batches = []
            topic_batches = []
            for a in algos:
                topics, items = a.sample_actions([u]) # recommend for user u using their current history 

                topic_batches.append(topics) 
                item_batches.append(items) 

            # action_batches = [a.sample_actions([context]).ravel() for a in algos] #(num_algos, rec_batch_size)
            print('  rec_topic: {}'.format(topic_batches))

            print('  rec_news: {}'.format(item_batches))

            reward_batches = [simulator.reward([u], items).ravel() for items in item_batches] #(num_algos, rec_batch_size)
            #@TODO: simulator has a complete history of each user, and it uses that complete history to simulate reward. 
            print('  rewards: {}'.format(reward_batches))

            h_items = np.concatenate((h_items, np.array(item_batches)[:,:,None]), axis=2)
            h_rewards = np.concatenate((h_rewards, np.array(reward_batches)[:,:,None]), axis=2)

            # Update the data buffer and clicked history
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
            [a.update(topics, items, rewards, mode = 'topic') for a in algos]

            if t % args.update_period == 0: # Update the item model (i.e. news_encoder and user_encoder)
                [a.update(topics, items, rewards, mode = 'item') for a in algos]

        h_items_all.append(h_items)
        h_rewards_all.append(h_rewards) # (n_trials, n_algos, rec_bs, T)

    return np.array(h_items_all), np.array(h_rewards_all)
