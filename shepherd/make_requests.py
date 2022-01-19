import os
import sys
import json
import requests

import gym
sys.path.append(os.path.abspath(__file__ + '/../..'))
import gym_envs


def send_json_to_website(d, path):
    """ Sends "d", a dictionary, to the website, and returns the response as a Python dictionary
    """
    r = requests.post('http://localhost:8000/'+ path, json=d)
    return r.json()

# Make the environment on client side (here, gym env)
env_name = 'LunarLander-v2'
env = gym.make(env_name)
obs = env.reset()

# First communication: login user
ok = send_json_to_website({'apikey': '5b9f9b0e-8006-4b76-ba15-e78b54bb060e'}, 'shepherd/login_user/')
print(ok)

if 'ok' in ok:
    print("Login successful")
    session_key = ok['session_key']

    # Start sending observations, in exhange of actions with the agent on the server side
    action = send_json_to_website({'obs': obs.tolist(), 'reward': 0.0, 'done': False, 'info': {}, 'session_key': session_key}, 'shepherd/env/')

    while True:
        print("ACTION ", action)
        obs, reward, done, info = env.step(action['action'])
        action = send_json_to_website({'obs': obs.tolist(), 'reward': reward, 'done': done, 'info': {}, 'session_key': session_key}, 'shepherd/env/')
        print(action) 
        # End of an episode, beginning of a new one
        if action['action'] is None:
            assert(done)
            obs = env.reset()
            action = send_json_to_website({'obs': obs.tolist(), 'reward': 0.0, 'done': False, 'info': {}, 'session_key': session_key}, 'shepherd/env/')

    # When the client wants to stop, its last communication is to return None as observation
    action = send_json_to_website({'obs': None}, 'shepherd/env/')
    
else:
    print("ERROR: Could not login to server")
