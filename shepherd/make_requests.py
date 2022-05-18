# This file is part of Shepherd.
#
# Copyright 2022 Hélène Plisnier
#
# Shepherd is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Shepherd is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along
# with Shepherd. If not, see <https://www.gnu.org/licenses/>.

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
