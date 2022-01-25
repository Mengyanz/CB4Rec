"""Define abstract CB algorithm. """

import numpy as np 
from utils.data_util import load_cb_train_data, load_cb_valid_data

class ContextualBanditLearner(object):
    def __init__(self, args, rec_batch_size = 1):
        self.args = args
        self.rec_batch_size =  rec_batch_size

    def sample_actions(self, user_samples):
        """Choose an action given a context. 
        
        Args:
            user_samples: A list of format (poss, negs, his, uid, tsp)

        Return: 
            top_k: top `rec_batch_size` news ids associated with each user_samples
        """
        pass 

    def update(self, contexts, h_actions, h_rewards, mode = 'learner'):
        """Update its internal model. 

        Args:
            context: list of user samples 
            h_actions: (num_context, rec_batch_size,) 
            h_rewards: (num_context, rec_batch_size,) 
            mode: one of {'learner', 'ts', 'user_emb'}
        """
        print('Abstract update for the internal model of the bandit!')
        pass

    def reset(self, seed):
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

    # TODO: in each round, sample user from selected users
    for e in range(args.n_trials):
        h_actions_all = [] 
        h_rewards_all = []

        print('trial = {}'.format(e))
        # independents runs to show empirical regret means, std

        # @TODO: Load the CB train data for this trial and pre-train each CB learner on this
        print('Load the CB train data for this trial and pre-train each CB learner on this')

        # Load the CB valid data for this trial 
        contexts = load_cb_valid_data(args, trial=e)
        num_contexts = len(contexts)

        
        # Run the contextual bandit process
        h_actions = np.empty((len(algos), rec_batch_size,0), float)
        h_rewards = np.empty((len(algos), rec_batch_size,0), float)
        for i in range(num_contexts):
            # iterate over selected users
            print('iteration', i)
            context = contexts[i] # user_sample
            action_batches = [a.sample_actions([context]).ravel() for a in algos] #(num_algos, rec_batch_size)
            print('Recommended actions: {}'.format(action_batches))

            reward_batches = [simulator.reward([context], action_batch).ravel() for action_batch in action_batches] #(num_algos, rec_batch_size)
            print('Rewards: {}'.format(reward_batches))
            # TODO: actions shoulds be recommended topics
            for j, a in enumerate(algos):
                a.update(contexts, action_batches[j], reward_batches[j], mode = 'ts')

            h_actions = np.concatenate((h_actions, np.array(action_batches)[:,:,None]), axis=2)
            h_rewards = np.concatenate((h_rewards, np.array(reward_batches)[:,:,None]), axis=2)
                

            for j, a in enumerate(algos):
                a.update(contexts, h_actions[j], h_rewards[j], mode = 'learner')

        h_actions_all.append(h_actions)
        h_rewards_all.append(h_rewards)

    # TODO: records all results for different exper and round
    return np.array(h_actions_all), np.array(h_rewards_all)
