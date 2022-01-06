import gym
from gym import spaces
from gym.utils import seeding
import queue
import random
import sys
import numpy as np
import time

def json_to_space(j):
    if isinstance(j, int):
        return spaces.Discrete(j)
    elif len(j) == 2:
        return spaces.Box(
            np.array(j[0]),
            np.array(j[1])
        )
    elif len(j) == 3:
        if isinstance(j[1], int):
            dtype = np.uint8   # Integer low, use uint8 (probably an image)
        else:
            dtype = np.float32 # Non-integer low, use float32

        return spaces.Box(
            shape=j[0],
            low=j[1],
            high=j[2],
            dtype=dtype
        )
    else:
        raise Exception("a space description must be an integer (discrete space), a tuple of (low, high), or a tuple of (shape, low, high)")

class ShepherdEnv(gym.Env):
    def __init__(self, parent_thread = None, observation_space = None, action_space = None):
        self.parent_thread = parent_thread
        self.observation_space = json_to_space(observation_space)
        self.action_space = json_to_space(action_space)
        self.time_when_i_got_an_observation = time.monotonic()
               
    def reset(self):
        """ Reset the environment and return the initial state number
        """
        return self.parent_thread.q_obs.get()[0]
        
    def step(self, action):
        now = time.monotonic()
        time_the_agent_spent_choosing_an_action = now - self.time_when_i_got_an_observation
        
        self.parent_thread.time_spent_in_agent += time_the_agent_spent_choosing_an_action
        
        self.parent_thread.q_actions.put(action)
        rs = self.parent_thread.q_obs.get()
        
        self.time_when_i_got_an_observation = time.monotonic()
        
        return rs

    def get_advice(self, obs):
        """ Return an advice torch.distributions.distribution.Distribution that
            encodes advice that the environment can provide for <obs>. For
            ShepherdEnv, self.parent_thread is asked for advice from the other
            agents in the same environment.
        """
        return self.parent_thread.get_advice(obs)
