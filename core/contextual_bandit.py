"""Define abstract CB algorithm. """

class ContextualBanditLearner(object):
    def __init__(self): 
        pass 

    def sample_action(self, contexts):
        """Choose an action given a context. """
        pass 

    def update(self, contexts, actions, rewards):
        """Update its internal model. """
        pass

    def reset(self, seed):
        pass 

