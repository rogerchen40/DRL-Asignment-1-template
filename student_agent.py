# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

def get_action(obs):
    #choose a random action
    #TODO: train your own agent
    return random.choice([0, 1, 2, 3, 4, 5])
