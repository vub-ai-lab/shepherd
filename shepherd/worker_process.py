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

def spawn_worker_process(to_process, from_process, action_space, observation_space, algorithm, algorithm_kwargs, log_name):
    p = multiprocessing.Process(target=worker_process_fun, args=
        (to_process, from_process, action_space, observation_space, algorithm, algorithm_kwargs, log_name)
    )

    p.start()
    return p

def worker_process_fun(to_process, from_process, action_space, observation_space, algorithm, algorithm_kwargs, log_name):
    # Spawn the worker thread
    q_obs = queue.Queue()
    q_act = queue.Queue()

    worker_thread = WorkerThread(q_obs, q_act, action_space, observation_space, algorithm, algorithm_kwargs, log_name)
    worker_thread.start()

    # Listen for request from the Shepherd server, and process them
    while True:
        request = to_process.get()

        if request["type"] == "advice":
            # The Shepherd server wants to get advice from the agent for an observation "obs"
            obs = request["obs"]
            response = {"advice": worker_thread.get_probas(obs)}
        elif request["type"] == "action":
            # The Shepherd server wants to get an action for an observation. Send it to the worker thread.
            # NOTE: request["obs_tuple"][0] is the observation (a dict), that contains advice :-) . The Shepherd server is responsible for producing advice and stuffing it in the observations.
            obs = request["obs_tuple"]
            response = {}

            if obs[2] == True:
                response["return"] = worker_thread.last_episode_return() + obs[1]  # Add the reward of the last time-step, not yet in last_episode_return

            q_obs.put(obs)
            response["action"] = q_act.get()
        elif request["type"] == "stop":
            # Stop this process
            return

        # Reply to the Shepherd server
        from_process.put(response)

class WorkerThread(threading.Thread):
    def __init__(self, q_obs, q_act, action_space, observation_space, algorithm, algorithm_kwargs, log_name):
        super().__init__()

        # Make the Gym environment
        self.env = ShepherdEnv(q_obs, q_act, action_space, observation_space)

        # Save a checkpoint every X steps
        self.save_name = log_name

        self.checkpoint_callback = CheckpointCallback(
            save_freq=algorithm_kwargs.get('save_freq', 1000),
            save_path='./logs_'+self.save_name+'/',
            name_prefix='rl_model'
        )

        # Instantiate the agent
        algo = ALGORITHMS[algorithm]
        self.learner = algo('MultiInputPolicy', self.env, verbose=1, **algorithm_kwargs)


    def last_episode_return(self):
        return self.env.episode_return

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

        return self.learner.get_advice(obs).cpu().numpy()

    def run(self):
        # Check if there is an already existing model to be loaded
        model = get_latest_model(self.save_name)

        if model != None:
            print("LOADING", model)

            try:
                self.learner.load(model)
            except:
                print("Unable to load saved model, the agent's config may have changed in the database")

        # Run the learner
        self.learner.learn(total_timesteps=100000000, callback = self.checkpoint_callback) # TODO while true instead of fixed number timesteps
        print('My thread died')
