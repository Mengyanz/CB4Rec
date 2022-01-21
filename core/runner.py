"""Define an abstract class for an interation model. """


class Runner(object): 
    def __init__(self, learner, simulator): 
        self.learner = learner 
        self.simulator = simulator 


    def run_one_iteration(self): 
        """Perform one iteration 
        - the learner sees the user 
        - the learner recommends top-K items 
        - simulator yields simulated rewards 
        - the learner updates its internal model. 
        """
        pass 