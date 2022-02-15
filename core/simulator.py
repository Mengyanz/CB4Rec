"""Define an abstract class for simulator. 

A simulator takes in a item id and a user id, and output a simulated reward in {0,1} 
"""
import numpy as np 
from collections import defaultdict

class Simulator(object): 
    def __init__(self, name='Simulator'): 
        """Contains an internal model that can be trained, evaluated, and deployed. 
        """
        self.name = name 
        print(name)
        self.reset()
        

    def set_clicked_history(self, init_clicked_history):
        self.clicked_history = init_clicked_history

    def update_data_buffer(self, pos, neg, uid, t): 
        self.data_buffer.append([pos, neg, self.clicked_history[uid], uid, t]) 

    def update_clicked_history(self, pos, uid): #TODO: think more if this is necessary for training simulator. 
        for item in pos:
            if item not in self.clicked_history[uid]:
                self.clicked_history[uid].append(item)

    def train(self): 
        """Train the internal model.
        """
        pass

    def reward(self, uid, news_indexes): 
        """Returns a simulated reward. 

        Args:
            uid: str, a user id   
            news_indexes: a list of item index (not nID, but its integer version)

        Return: 
            rewards: (n,m) of 0 or 1 
        """
        pass 

    def reset(self):
        """Reset the simulator to its initial state. """
        self.clicked_history = defaultdict(list) # a dict, key: user, value: clicked history of a user at current time 
        self.data_buffer = [] # a list of [pos, neg, hist, uid, t] samples collected  
