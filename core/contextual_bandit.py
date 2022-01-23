"""Define abstract CB algorithm. """

import numpy as np 

class ContextualBanditLearner(object):
    def __init__(self, rec_batch_size = 1):
        self.rec_batch_size =  rec_batch_size

    def sample_actions(self, user_samples):
        """Choose an action given a context. 
        
        Args:
            user_samples: A list of format (poss, negs, his, uid, tsp)

        Return: 
            top_k: top `rec_batch_size` news ids associated with each user_samples
        """
        pass 

    def update(self, context, action_batch, reward_batch):
        """Update its internal model. 

        Args:
            context: a user sample. 
            action_batch: (rec_batch_size,) 
            reward_batch: (rec_batch_size,) 
        """
        print('Updating the internal model of the bandit!')
        pass

    def reset(self, seed):
        pass 


def run_contextual_bandit(contexts, simulator, rec_batch_size, algos):
    """Run a contextual bandit problem on a set of algorithms.
    Args:
        contexts: A list of user samples. 
        simulator: An instance of Simulator.
        rec_batch_size: int, number of recommendations per context.
        algos: List of algorithms (instances of `ContextualBanditLearner`) to use in the contextual bandit instance.
    Returns:
        h_actions: Matrix with actions: size (num_context, batch_size, num_algorithms).
        h_rewards: Matrix with rewards: size (num_context, num_algorithms).
    """

    num_contexts = len(contexts)

    h_actions = np.empty((len(algos), rec_batch_size,0), float)
    h_rewards = np.empty((len(algos), rec_batch_size, 0), float)

    # Run the contextual bandit process
    for i in range(num_contexts):
        print('iteration', i)
        context = contexts[i] # user_sample
        action_batches = [a.sample_actions([context]).ravel() for a in algos] #(num_algos, rec_batch_size)
        print(action_batches)

        reward_batches = [simulator.reward([context], action_batch).ravel() for action_batch in action_batches] #(num_algos, rec_batch_size)
        print(reward_batches)

        for j, a in enumerate(algos):
            a.update(context, action_batches[j], reward_batches[j])

        h_actions = np.concatenate((h_actions, np.array(action_batches)[:,:,None]), axis=2)
        h_rewards = np.concatenate((h_rewards, np.array(reward_batches)[:,:,None]), axis=2)

    return h_actions, h_rewards
