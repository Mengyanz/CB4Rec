"""Define abstract CB algorithm. """

import numpy as np 

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

    def update(self, contexts, h_actions, h_rewards):
        """Update its internal model. 

        Args:
            context: list of user samples 
            h_actions: (num_context, rec_batch_size,) 
            h_rewards: (num_context, rec_batch_size,) 
        """
        print('Abstract update for the internal model of the bandit!')
        pass

    def reset(self, seed):
        pass 


def run_contextual_bandit(args, contexts, simulator, rec_batch_size, algos):
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

    contexts = contexts[:3]

    num_contexts = len(contexts)
    num_exper = args.num_exper
    num_round = args.num_round

    # TODO: in each round, sample user from selected users
    for e in range(num_exper):
        # independents runs to show empirical regret means, std
        for j in range(num_round):
            # Run the contextual bandit process
            h_actions = np.empty((len(algos), rec_batch_size,0), float)
            h_rewards = np.empty((len(algos), rec_batch_size,0), float)
            for i in range(num_contexts):
                # iterate over selected users
                print('iteration', i)
                context = contexts[i] # user_sample
                action_batches = [a.sample_actions([context]).ravel() for a in algos] #(num_algos, rec_batch_size)
                print(action_batches)

                reward_batches = [simulator.reward([context], action_batch).ravel() for action_batch in action_batches] #(num_algos, rec_batch_size)
                print(reward_batches)

                h_actions = np.concatenate((h_actions, np.array(action_batches)[:,:,None]), axis=2)
                h_rewards = np.concatenate((h_rewards, np.array(reward_batches)[:,:,None]), axis=2)

            for j, a in enumerate(algos):
                a.update(contexts, h_actions[j], h_rewards[j])

    # TODO: records all results for different exper and round
    return h_actions, h_rewards
