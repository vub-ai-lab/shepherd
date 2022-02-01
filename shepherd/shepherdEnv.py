import gym
import numpy as np

def json_to_space(j):
    if isinstance(j, int):
        return gym.spaces.Discrete(j)
    elif isinstance(j, dict):
        return gym.spaces.Dict({
            k: json_to_space(v) for k, v in j.items()
        })
    elif isinstance(j, list) and len(j) == 3:
        if isinstance(j[1], int):
            dtype = np.uint8   # Integer low, use uint8 (probably an image)
        else:
            dtype = np.float32 # Non-integer low, use float32

        return gym.spaces.Box(
            shape=tuple(j[0]),
            low=j[1],
            high=j[2],
            dtype=dtype
        )
    else:
        raise Exception("a space description must be an integer (discrete space), a list of (shape, low, high), or a dictionary of keys to integers or lists.")

class ShepherdEnv(gym.Env):
    def __init__(self, q_obs, q_act, action_space, observation_space):
        self.action_space = json_to_space(action_space)
        self.observation_space = json_to_space(observation_space)
        self.q_obs = q_obs
        self.q_act = q_act

        # Compute the advice shape
        if isinstance(self.action_space, gym.spaces.Discrete):
            # Discrete actions, multinomial advice
            advice_shape = (self.action_space.n,)
        elif isinstance(self.action_space, gym.spaces.Box):
            # Continuous actions, mean/std advice
            advice_shape = (2,) + self.action_space.shape
        else:
            raise Exception("Actions must be discrete or continuous, not a dictionary")

        # Add advice to the observation
        if not isinstance(self.observation_space, gym.spaces.Dict):
            # Force MultiInput observations, even if the environment provides a single observation
            self.observation_space = gym.spaces.Dict({
                "obs": self.observation_space
            })

        self.observation_space.spaces["advice"] = gym.spaces.Box(shape=advice_shape, low=0.0, high=1.0)
        print("Observation space", self.observation_space)
               
    def reset(self):
        """ Reset the environment and return the initial state
        """
        self.episode_return = 0.0

        return self.get_state()[0]
        
    def step(self, action):
        self.q_act.put(action)
        s = self.get_state()

        self.episode_return += s[1]  # state, [reward], done, info

        if s[2]:
            # The episode finished
            # Emit a "None" action, that allows to balance the number of writes to q_act and reads of q_obs.
            # Reset will read from q_obs but will not write to q_act
            self.q_act.put(None)

        return s

    def get_state(self):
        """ Get a state, reward, done, info tuple from the queue
        """
        s, r, d, i = self.q_obs.get()
        return s, r, d, i
