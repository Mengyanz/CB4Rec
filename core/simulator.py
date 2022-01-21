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

    def simulate(self, news_id, history_id): 
        """Returns a simulated reward. 

        Args:
            news_id: int, news index
            history_id: int, history index representing a user

        Return: 
            0 or 1 
        """
        pass 