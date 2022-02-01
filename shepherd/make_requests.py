import os
import sys
import json
import requests

import gym
import numpy as np
sys.path.append(os.path.abspath(__file__ + '/../..'))
import gym_envs


def send_json_to_website(d, path):
    """ Sends "d", a dictionary, to the website, and returns the response as a Python dictionary
    """
    r = requests.post('http://localhost:8000/'+ path, json=d)
    return r.json()

def obs_to_json(obs):
    if isinstance(obs, int):
        return obs
    elif isinstance(obs, np.ndarray):
        return obs.tolist()
    elif isinstance(obs, dict):
        return {k: obs_to_json(v) for k, v in obs.items()}

# Make the environment on client side (here, gym env)
env_name = sys.argv[2]
env = gym.make(env_name)

# First communication: login user
apikey = sys.argv[1]
ok = send_json_to_website({'apikey': apikey}, 'shepherd/login_user/')
print(ok)

if 'ok' in ok:
    print("Login successful")
    session_key = ok['session_key']

    while True:
        obs = env.reset()
        done = False
        reward = 0.0

        while True:
            # Ask Shepherd for an action
            action = send_json_to_website({'obs': obs_to_json(obs), 'reward': reward, 'done': done, 'info': {}, 'session_key': session_key}, 'shepherd/env/')
            print(action)

            if action['action'] is None:
                break

            # Ask the environment for an observation
            obs, reward, done, info = env.step(action['action'])
else:
    print("ERROR: Could not login to server")
