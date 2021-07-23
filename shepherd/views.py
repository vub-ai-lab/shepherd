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
import os
from .models import Agent, Algorithm, Parameter, ParameterValue
import gym
from gym import spaces

sys.path.insert(0, os.path.abspath(__file__ + '/../..'))
sys.path.insert(0, os.path.abspath(__file__ + '/../../stable-baselines3/'))
sys.path.insert(0, os.path.abspath(__file__ + '/../../stable-baselines3-contrib/'))

print(sys.path)

import gym_envs

from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from sb3_contrib import BDPI
from stable_baselines3.common.callbacks import CheckpointCallback

known_threads = {}
known_threads_id = 0

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

def clean_up_logs(directory):
    """
    Remove old logs and leaves only the latest .zip log file of a RL model
    """
    logs = glob.glob(directory + '/*') # * means all if need specific format then *.csv
    latest_log = max(logs, key=os.path.getctime)
    logs.remove(latest_log)
    for log in logs:
        os.remove(log)


def get_latest_log(directory):
    try:
        latest_log = max(glob.glob(directory + '/*'), key=os.path.getctime)
    except ValueError:
        latest_log = None
    
    return latest_log


class ShepherdThread(threading.Thread):
    def __init__(self, observation_space, action_space, agent, known_agent_id):
        super().__init__()
        
        self.q_obs = queue.Queue()
        self.q_actions = queue.Queue()
        self.advisors = []
        self.known_agent_id = known_agent_id
        self.agent_id = agent.id # each agent has a bunch of threads that can be used as each other's advisors
        self.cumulative_reward = 0.0
        self.env = gym.make("ShepherdEnv-v0", parent_thread = self, observation_space = observation_space, action_space = action_space)
        
        # Get parameter values        
        kwargs = get_param_values_from_database(agent)
        print("KWARGS ", kwargs)

        # Save a checkpoint every X steps
        self.save_name = str(agent.owner) + '_' + agent.algo.name + '_' + str(agent.id)
        self.checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs_'+self.save_name+'/', name_prefix='rl_model'+'_'+str(self.known_agent_id))
        
        self.learner = eval(agent.algo.name+'("MlpPolicy", env, verbose=1, **kwargs)', {}, {agent.algo.name: algorithms[agent.algo.name], "env": self.env, "kwargs": kwargs})
        

    def updateAdvisorsList(self, advisors):
        for advisor in advisors:
            if advisor not in self.advisors and advisor.agent_id == self.agent_id and advisor != self:
                # a thread can only be advised by other threads with the same agent_id
                self.advisors.append(advisor)
                self.learner.add_advisor(advisor.learner)
        for advisor in self.advisors:
            if advisor not in advisors:
                self.advisors.remove(advisor)
                self.learner.remove_advisor(advisor.learner)


    def run(self):
        # Check if there is an already existing model to be loaded
        zip_file = get_latest_log('./logs_'+self.save_name+'/')
        if zip_file != None:
            print("LOADING")
            self.learner.load(zip_file)
        # Run the learner
        self.learner.learn(total_timesteps=1000000, callback = self.checkpoint_callback) # TODO while true instead of fixed number timesteps
        print('My thread died')
        return None        
        

def action_to_json(a):
    if isinstance(a, np.ndarray):
        return a.tolist()
    else:
        return int(a)


@csrf_exempt 
def login_user(request):
    
    global known_threads, known_threads_id, cumulative_reward
    
    data = json.loads(request.body) 
    
    # login and find the user's agent
    user = authenticate(request, username=data['username'], password=data['user_password'])
    try:
        login(request, user)
        agent = Agent.objects.get(owner=user, id=data['agent_id'])
    except AttributeError:
        print("ERROR: Could not login user")
        return JsonResponse({'ok': False}, safe=False)
    except Agent.DoesNotExist:
        print("ERROR: Could not find agent in the user's agents")
        return JsonResponse({'ok': False}, safe=False)
    
    # Create a thread for the agent
    agent_thread = ShepherdThread(str_to_json(agent.observation_space), str_to_json(agent.action_space), agent, known_threads_id)
    agent_thread.updateAdvisorsList(list(known_threads.values()))
    known_threads[known_threads_id] = agent_thread
    request.session['known_thread_id'] = known_threads_id
    known_threads_id+=1
    
    agent_thread.start()
        
    return JsonResponse({'ok': True}, safe=False)


@csrf_exempt
@login_required
def env(request):
    
    global known_threads
    
    data = json.loads(request.body) 
    
    # Find the thread 
    known_thread_id = request.session['known_thread_id']
    try:
        thread = known_threads[known_thread_id]
    
    except KeyError:
        print("ERROR: Could not find thread in the current threads pool")
        return None # TODO ???
    
    # Update its individual advisors list if new advisors were recently added
    thread.updateAdvisorsList(list(known_threads.values()))
    print(str(thread.getName()), ' has ADVISORS ', thread.advisors)
    
    # Put the observation and reward into the agent
    # if obs is None, the client wanst to end communication
    if data['obs'] == None:
        thread.q_obs.put(None)
        del known_threads[known_thread_id] # TODO Not sure we need to ever delete them..?
        return JsonResponse({'action': None}, safe=False)
    
    thread.q_obs.put((
       np.array(data['obs']), data['reward'], data['done'], data['info']
    ))
        
    # log rewards in out file to plot learning curves    
    thread.cumulative_reward += data['reward']
    if data['done']:
        f = open('out-'+str(known_thread_id), 'a')
        print(thread.cumulative_reward, file=f)
        thread.cumulative_reward = 0.0
        f.flush()

    # The learner will have produced an action
    # if done=True, the episode has ended and the environment on the client side must be reset for a new episode to begin
    if data['done']:
        action = None
    else:
        action = action_to_json(thread.q_actions.get())
        

    return JsonResponse({'action': action}, safe=False)
 
    
    
