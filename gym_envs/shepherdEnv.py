import gym
from gym import spaces
from gym.utils import seeding
import queue
import random
import sys
import numpy as np


def json_to_space(j):
    if isinstance(j, int):
        return spaces.Discrete(j)
    else:
        return spaces.Box(
            np.array(j[0]),
            np.array(j[1])
        )

class ShepherdEnv(gym.Env):
    def __init__(self, parent_thread = None, observation_space = None, action_space = None):
        self.parent_thread = parent_thread
        self.observation_space = json_to_space(observation_space)
        self.action_space = json_to_space(action_space)
               
    def reset(self):
        """ Reset the environment and return the initial state number
        """
        return self.parent_thread.q_obs.get()[0]
        
    def step(self, action):
        self.parent_thread.q_actions.put(action)
        return self.parent_thread.q_obs.get()
        
