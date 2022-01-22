"""Define an abstract class for simulator. 

A simulator takes in a item id and a user id, and output a simulated reward in {0,1} 
"""

class Simulator(object): 
    def __init__(self): 
        """Contains an internal model that can be trained, evaluated, and deployed. 
        """
        print('Hello')
        pass 

    def train(sefl): 
        """Train the internal model.
        """

    def reward(self, user_samples, news_ids): 
        """Returns a simulated reward. 

        Args:
            news_id: a list of n int, news ids. 
            user_samples: a list of m user samples

        Return: 
            rewards: (n,m) of 0 or 1 
        """
        pass 