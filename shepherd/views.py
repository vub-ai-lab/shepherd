from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User # import the database classes
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
import json

import threading
import queue
import numpy as np
import ast
import sys
import glob
import time
import os
from .models import *
import gym
import torch as th
from gym import spaces

sys.path.insert(0, os.path.abspath(__file__ + '/../..'))
sys.path.insert(0, os.path.abspath(__file__ + '/../../stable-baselines3/'))
sys.path.insert(0, os.path.abspath(__file__ + '/../../stable-baselines3-contrib/'))

import gym_envs

from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from sb3_contrib import BDPI
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import avg_distributions

ALL_THREADS = []
THREAD_POOLS = {}
LOCK = threading.Lock()

algorithms = {'BDPI': BDPI, 'A2C': A2C, 'DDPG': DDPG, 'DQN': DQN, 'HER': HER, 'PPO': PPO, 'SAC': SAC, 'TD3': TD3}

def str_to_json(space):
    try:
        space = int(space)
    except ValueError:
        return ast.literal_eval(space)
    if isinstance(space, int):
        return space
    else:
        print("ERROR, space is neither an int, nor a list of high and low")
        return None
    
def get_param_values_from_database(agent):
    """
    Retrieve parameter value for that agent, with that algorithm, either set by the user or the default value of the parameter
    """
    params = list(Parameter.objects.filter(algo=agent.algo))
    param_values = list(ParameterValue.objects.filter(agent=agent))
    
    arguments = {}
    
    for param in params:
        arguments[param.name] = param
        for i in range(len(param_values)):
            if param_values[i].param == param:
                arguments[param.name] = param_values[i]
            
        if param.t == 1:
            arguments[param.name] = arguments[param.name].value_bool
        elif param.t == 2:
            arguments[param.name] = arguments[param.name].value_int
        elif param.t == 3:
            arguments[param.name] = arguments[param.name].value_float
        else:
            arguments[param.name] = arguments[param.name].value_str
            
    return arguments

def get_latest_log(directory):
    try:
        latest_log = max(glob.glob(directory + '/*'), key=os.path.getctime)
    except ValueError:
        latest_log = None
    
    return latest_log

class AgentThreadPool:
    """ List of ShepherdThread instances for a particular agent_id. Used to produce advice
    """

    def __init__(self):
        self.threads = []

    def add_thread(self, t):
        """ Add a thread to the list of threads
        """
        self.threads.append(t)

    def cleanup_threads(self):
        """ Explore the threads and remove those that have not seen any recent activity.
        """
        i = 0
        current_time = time.monotonic()

        while i < len(self.threads):
            if (current_time - self.threads[i].last_activity_time) > 30*60:
                # No activity since 30 minutes
                del self.threads[i]
            else:
                i += 1

    def get_advice(self, obs, current_thread):
        """ Ask all the threads to produce an advice vector for observation obs
        """
        action_space = self.threads[0].env.action_space

        if isinstance(action_space, spaces.Discrete):
            # Advice current_thread with every thread that has a last_activity_time
            # later than the start_time of current_thread
            threads = []

            for t in self.threads:
                if (t is not current_thread) and (t.last_activity_time > current_thread.start_time):
                    threads.append(t)

            if len(threads) > 1:
                advice_distributions = [a.get_probas(obs).distribution for a in threads]
                return avg_distributions(advice_distributions, adviceact=0.8)
            else:
                # No advisor (only one thread, ourselves). Return a null distribution
                return th.distributions.categorical.Categorical(logits=th.ones(action_space.n,))
        else:
            raise NotImplementedError("Continuous-action environments are not supported by Shepherd yet")


class ShepherdThread(threading.Thread):
    def __init__(self, observation_space, action_space, agent, known_agent_id, pool):
        super().__init__()
        
        self.q_obs = queue.Queue()
        self.q_actions = queue.Queue()
        self.pool = pool

        self.agent = agent
        self.cumulative_reward = 0.0
        self.start_time = time.monotonic()
        self.last_activity_time = time.monotonic()

        self.env = gym.make("ShepherdEnv-v0", parent_thread = self, observation_space = observation_space, action_space = action_space)
        
        # Get parameter values        
        kwargs = get_param_values_from_database(agent)
        print("KWARGS ", kwargs)

        # Save a checkpoint every X steps
        self.save_name = str(agent.owner) + '_' + agent.algo.name + '_' + str(agent.id)
        self.checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs_'+self.save_name+'/', name_prefix='rl_model'+'_'+str(known_agent_id))
        
        # Instantiate the agent
        algo = algorithms[agent.algo.name]

        self.learner = algo(agent.policy, self.env, verbose=1, **kwargs)

    def get_probas(self, obs):
        """ Ask the learner for action probabilities for observation <obs>
        """
        return self.learner.policy.get_probas(obs)

    def get_advice(self, obs):
        """ Ask the threadpool for advice
        """
        return self.pool.get_advice(obs, self)

    def run(self):
        # Check if there is an already existing model to be loaded
        zip_file = get_latest_log('./logs_'+self.save_name+'/')
        if zip_file != None:
            print("LOADING", zip_file)
            self.learner.load(zip_file)

        # Run the learner
        self.learner.learn(total_timesteps=100000000, callback = self.checkpoint_callback) # TODO while true instead of fixed number timesteps
        print('My thread died')
        return None        
        

def action_to_json(a):
    if isinstance(a, np.ndarray):
        return a.tolist()
    else:
        return int(a)

def wrap_response(response):
    """ This is what allows a Shepherd server to answer JSON queries from Javascript pages on different servers.
    """
    response["Access-Control-Allow-Origin"] = "https://steckdenis.be"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response["Access-Control-Max-Age"] = "1000"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"

    return response

@csrf_exempt 
def login_user(request):
    global ALL_THREADS, THREAD_POOLS, LOCK
    
    data = json.loads(request.body) 

    # Look for an API key
    try:
        key = APIKey.objects.select_related('agent').get(key=data['apikey'])
        agent = key.agent
    except AttributeError:
        return wrap_response(JsonResponse({'error': 'JSON query data must contain an apikey element'}))
    except APIKey.DoesNotExist:
        return wrap_response(JsonResponse({'error': 'API Key not found in the database'}))
    
    # Create a pool for the agent_id if necessary
    agent_id = agent.id

    with LOCK:
        if agent_id not in THREAD_POOLS:
            THREAD_POOLS[agent_id] = AgentThreadPool()

        thread_pool = THREAD_POOLS[agent_id]

        # Create a thread for the agent
        thread_id = len(ALL_THREADS)
        agent_thread = ShepherdThread(str_to_json(agent.observation_space), str_to_json(agent.action_space), agent, thread_id, thread_pool)

        thread_pool.add_thread(agent_thread)
        ALL_THREADS.append(agent_thread)

    request.session['thread_id'] = thread_id
    request.session['agent_id'] = agent_id
    agent_thread.start()

    request.session.create()

    return wrap_response(JsonResponse({'ok': True, 'session_key': request.session.session_key}))


@csrf_exempt
def env(request):
    global ALL_THREADS
    
    data = json.loads(request.body) 

    # Load the session from the session_key in the request
    Store = type(request.session)

    try:
        session = Store(session_key=data['session_key'])
    except:
        raise
        return wrap_response(JsonResponse({'error': "Unable to reload session, is session_key sent as JSON in the request?"}))
    
    # Find the thread 
    thread_id = session['thread_id']
    agent_id = session['agent_id']

    try:
        thread = ALL_THREADS[thread_id]
    except KeyError:
        return wrap_response(JsonResponse({'error': "Could not find thread in the current threads pool"}))
    
    # Send the observation to the thread
    thread.q_obs.put((
       np.array(data['obs']), data['reward'], data['done'], data['info']
    ))

    # Update the activity time of the thread, and cleanup stale threads
    thread.last_activity_time = time.monotonic()

    THREAD_POOLS[agent_id].cleanup_threads()
        
    # log episode returns to plot learning curves
    thread.cumulative_reward += data['reward']

    if data['done']:
        log_entry = EpisodeReturn()
        log_entry.agent = thread.agent
        log_entry.ret = thread.cumulative_reward
        log_entry.save()

        # Reset cumulative_reward for the next episode
        thread.cumulative_reward = 0.0

    # The learner will have produced an action
    # if done=True, the episode has ended and the environment on the client side must be reset for a new episode to begin
    if data['done']:
        action = None
    else:
        action = action_to_json(thread.q_actions.get())

    return wrap_response(JsonResponse({'action': action}))
 
    
    
