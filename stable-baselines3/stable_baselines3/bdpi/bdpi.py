from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.bdpi.policies import BDPIPolicy


class BDPI(OffPolicyAlgorithm):
    """
    Bootstrapped Dual Policy Iteration

    Sample-efficient discrete-action RL algorithmn, built on one actor trained
    to imitate the greedy policy of several Q-Learning critics.

    Paper: https://arxiv.org/abs/1903.04193

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor)
        it can be a function of the current progress remaining (from 1 to 0)
    :param actor_lr: Conservative Policy Iteration learning rate for the actor (used in a formula, not for Adam gradient steps)
    :param critic_lr: Q-Learning "alpha" learning rate for the critics
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[BDPIPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        actor_lr: float = 0.05,
        critic_lr: float = 0.2,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 256,
        batch_size: int = 256,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 20,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super(BDPI, self).__init__(
            policy,
            env,
            BDPIPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            0.0,
            gamma,
            train_freq,
            gradient_steps,
            None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=False,
            sde_sample_freq=1,
            use_sde_at_warmup=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete),
            sde_support=False
        )

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(BDPI, self)._setup_model()

        self.actor = self.policy.actor
        self.criticsA = self.policy.criticsA
        self.criticsB = self.policy.criticsB

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer] + [c.optimizer for c in self.criticsA] + [c.optimizer for c in self.criticsB]
        self._update_learning_rate(optimizers)

        actor_losses, critic_losses = [], []
        mse_loss = th.nn.MSELoss(reduction='sum')

        # Update every critic (and the actor after each critic)
        for criticA, criticB in zip(self.criticsA, self.criticsB):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Update the critic (code taken from DQN)
            with th.no_grad():
                qvA = criticA(replay_data.next_observations)
                qvB = criticB(replay_data.next_observations)
                qv = th.min(qvA, qvB)

                QN = th.arange(replay_data.next_observations.shape[0])
                next_q_values = qv[QN, qvA.argmax(1)].reshape(-1, 1)

                # 1-step TD target
                target_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Make real supervised learning target Q-Values (even for non-taken actions)
                target_q_values = criticA(replay_data.observations)
                actions = replay_data.actions.long().flatten()
                target_q_values[QN, actions] += self.critic_lr * (target_values.flatten() - target_q_values[QN, actions])

            for i in range(gradient_steps):
                predicted_q_values = criticA(replay_data.observations)
                loss = mse_loss(predicted_q_values, target_q_values)

                criticA.optimizer.zero_grad()
                loss.backward()
                criticA.optimizer.step()

            logger.record("train/critic_loss", float(loss.item()))
            logger.record("train/avg_q", float(target_q_values.mean()))

            # Update the actor
            with th.no_grad():
                greedy_actions = target_q_values.argmax(1)

                train_probas = th.zeros_like(target_q_values)
                train_probas[QN, greedy_actions] = 1.0

                # Normalize the direction to be pursued
                train_probas /= 1e-6 + train_probas.sum(1)[:, None]
                actor_probas = self.actor(replay_data.observations)

                # Imitation learning (or distillation, or reward-penalty Pursuit, all these are the same thing)
                alr = self.actor_lr
                train_probas = (1. - alr) * actor_probas + alr * train_probas
                train_probas /= train_probas.sum(-1, keepdim=True)


            for i in range(gradient_steps):
                predicted_probas = self.actor(replay_data.observations)
                loss = mse_loss(predicted_probas, train_probas)

                self.actor.optimizer.zero_grad()
                loss.backward()
                self.actor.optimizer.step()

            logger.record("train/actor_loss", float(loss.item()))

        # Swap QA and QB
        self.criticsA, self.criticsB = self.criticsB, self.criticsA

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "BDPI",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(BDPI, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
