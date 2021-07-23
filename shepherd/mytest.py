
import gym
import gym_envs



env_name = 'Table-v0'
env = gym.make(env_name)
print(env.observation_space)
print(env.action_space)
