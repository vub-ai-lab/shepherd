import time
import sys
import collections
import random
import datetime
import torch

from stable_baselines3.common.policies import mix_distributions

# Utility functions
def sample_wr(population, k):
    """ Chooses k random elements (with replacement) from a population
    """
    n = len(population)

    _random, _int = random.random, int  # speed hack
    result = [None] * k

    for i in range(k):
        j = _int(_random() * n)
        result[i] = population[j]

    return result

class TabularBDPIActor:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        self.probas = torch.ones((self.num_states, self.num_actions)) / self.num_actions

    def get_probas(self, state):
        return torch.distributions.categorical.Categorical(probs=self.probas[state])

class TabularBDPI:
    def __init__(self, policy, env, verbose, **kwargs):
        # Same parameters as normal sb3_contrib algorithms, but policy and verbose are ignored
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        self.env = env

        self.clr = kwargs.get('critic_lr', 0.1)
        self.alr = kwargs.get('actor_lr', 0.1)
        self.num_critics = kwargs.get('num_critics', 8)
        self.q_loops = kwargs.get('q_loops', 4)
        self.batch_size = kwargs.get('batch_size', 128)

        # BDPI critics and actor, in tabular form
        # - The critics are randomly initialized with values close to 0
        # - The actor is initialized to the uniform probability distribution
        self.critics = torch.rand((self.num_critics, self.num_states, self.num_actions)) * 0.01
        self.policy = TabularBDPIActor(self.num_states, self.num_actions)

        self.transitions = collections.deque([], 1000)
        self.critic_indexes = list(range(self.num_critics))

    def learn(self, total_timesteps, callback):
        done = False
        state = self.env.reset()

        for i in range(total_timesteps):
            # Ask for an action and advice
            advice_dist = self.env.get_advice(state)
            actor_dist = self.policy.get_probas(state)

            dist = mix_distributions(actor_dist, advice_dist)
            action = dist.sample()

            # Execute the action
            next_state, reward, done, _ = self.env.step(action)

            self.transitions.append([state, action, reward, next_state, done])

            # LEARN: Loop over critics
            random.shuffle(self.critic_indexes)

            for critic_index in self.critic_indexes:
                qtable = self.critics[critic_index]
                actor = self.policy.probas
                batch = sample_wr(self.transitions, self.batch_size)

                # Loop over q-loops
                for qloop in range(self.q_loops):
                    # Loop over experiences
                    for (s, a, r, ns, t) in batch:
                        vnext = qtable[ns].max()

                        td_error = r + (0.99 * vnext if not t else 0.0) - qtable[s, a]
                        qtable[s, a] += self.clr * td_error

                # Update the actor
                for (s, a, r, ns, t) in batch:
                    greedy_action = int(qtable[s].argmax())
                    target_probas = torch.zeros((self.num_actions,))
                    target_probas[greedy_action] = 1.0

                    actor[s] = (1. - self.alr) * actor[s] + self.alr * target_probas

            # Next time-step
            state = next_state
