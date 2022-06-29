# This file is part of Shepherd.
#
# Copyright 2022 the VU Brussels AI Lab, applied AI and consulting department, ai.vub.ac.be
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

import multiprocessing
import threading
import queue
import sys
import os
import glob

sys.path.insert(0, os.path.abspath(__file__ + '/../../stable-baselines3/'))

from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.evaluation import evaluate_policy


from .shepherdEnv import ShepherdEnv

ALGORITHMS = {'A2C': A2C, 'DDPG': DDPG, 'DQN': DQN, 'HER': HER, 'PPO': PPO, 'SAC': SAC, 'TD3': TD3}

"""
Architecture of the Shepherd worker processes:

- Each Agent has a pool of worker processes
- Logged-in clients get assigned a process in the pool (created if needed). There is one client per process at most.
- The processes run stable-baselines3 and produce advice

Each process spawns two threads:

- The worker thread runs agent.learn(), gets observations, and produces actions
- The communication thread manages the pipes to the Shepherd server, so that observations can be sent to the worker thread, and requests for advice can directly be processed by the agent's policy.
"""

def get_latest_model(save_name):
    directory = './logs_'+save_name+'/'
    latest_model = None
    try:
        latest_model = max(glob.glob(directory + '/*'), key=os.path.getctime)
    except ValueError:
        pass

    return latest_model

def spawn_worker_process(queues, action_space, observation_space, algorithm, algorithm_kwargs, log_name, evaluate):
    p = multiprocessing.Process(target=worker_process_fun, args=
        (queues, action_space, observation_space, algorithm, algorithm_kwargs, log_name, evaluate)
    )

    p.start()
    return p

def worker_process_fun(queues, action_space, observation_space, algorithm, algorithm_kwargs, log_name, evaluate):
    # Spawn the worker thread, that does the learning
    worker_thread = WorkerThread(queues["act_obs"], queues["act_rsp"], action_space, observation_space, algorithm, algorithm_kwargs, log_name, evaluate)
    worker_thread.start()

    # This (main) thread can now reply to requests for advice
    adv_obs = queues["adv_obs"]
    adv_rsp = queues["adv_rsp"]

    while True:
        obs = adv_obs.get()

        if obs is None:
            # Stop this process
            return
        else:
            # The Shepherd server wants to get advice from the agent for an observation "obs"
            response = worker_thread.get_probas(obs)

        # Reply to the Shepherd server
        adv_rsp.put(response)

class WorkerThread(threading.Thread):
    def __init__(self, q_obs, q_act, action_space, observation_space, algorithm, algorithm_kwargs, log_name, evaluate):
        super().__init__()

        # Make the Gym environment
        self.env = ShepherdEnv(q_obs, q_act, action_space, observation_space)

        # Save a checkpoint every X steps
        self.save_name = log_name
        self.evaluate = evaluate

        self.checkpoint_callback = CheckpointCallback(
            save_freq=algorithm_kwargs.get('save_freq', 1000),
            save_path='./logs_'+self.save_name+'/',
            name_prefix='rl_model'
        )

        # Instantiate the agent
        algo = ALGORITHMS[algorithm]
        self.learner = algo('MultiInputPolicy', self.env, verbose=1, **algorithm_kwargs)

    def get_probas(self, obs):
        """ Ask the learner for action probabilities for observation <obs>
        """
        # Cast obs to PyTorch tensors, and add a (1,) batch dimension to every tensor
        def add_batch(o):
            if isinstance(o, dict):
                return {k: add_batch(v) for k, v in o.items()}
            else:
                return o[None, ...]

        obs = obs_as_tensor(obs, self.learner.device)
        obs = add_batch(obs)

        return self.learner.get_advice(obs).cpu().numpy()[0]

    def run(self):
        # Check if there is an already existing model to be loaded
        model = get_latest_model(self.save_name)

        if model != None:
            print("LOADING", model)

            try:
                self.learner.load(model)
            except:
                print("Unable to load saved model, the agent's config may have changed in the database")

        if self.evaluate:
            print("Evaluating policy")
            evaluate_policy(self.learner, self.env, n_eval_episodes=100000000, return_episode_rewards=False)
        else:
            # Run the learner
            self.learner.learn(total_timesteps=100000000, callback = self.checkpoint_callback) # TODO while true instead of fixed number timesteps
            print('My thread died')
