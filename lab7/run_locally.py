#!/usr/bin/env python3

import argparse

from misio.util import generate_deterministic_seeds

from tqdm import trange
import gym
import numpy as np
import random
import time
import numpy as np
from ddpg import doDDPG
# -----------

random_state = np.random.RandomState(123)

def playGame(actor):
    random_state = np.random.RandomState(123)
    n_games = 10
    seed = 123
    seeds = generate_deterministic_seeds(seed, n_games)

    env = gym.make("Pendulum-v0")
    # To use
    # env = gym.make("BipedalWalker-v3")
    # agent = doDDPG()

    rewards = []
    for i_episode in trange(n_games):
        np.random.seed(seeds[i_episode])
        random.seed(seeds[i_episode])
        env.seed(int(seeds[i_episode]))
        state = env.reset()
        done = False
        total_reward = 0

        cur_state = env.reset()
        while not done:
            env.render()

            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            action = actor.act(cur_state)
            # print(action)

            new_state, reward, done, info = env.step(action)   
            total_reward += reward

            if done:
                print(env)
                break

        rewards.append(total_reward)

    print("Avg reward: {:0.2f}".format(np.mean(rewards)))
    print("Min reward: {:0.2f}".format(np.min(rewards)))
    print("Max reward: {:0.2f}".format(np.max(rewards)))
    print()
    env.close()


if __name__ == "__main__":

    n_games = 100
    seed = 123
    seeds = generate_deterministic_seeds(seed, n_games)

    env = gym.make("Pendulum-v0")
    # To use
    # env = gym.make("BipedalWalker-v3")



    rewards = []
    for i_episode in trange(n_games):
        np.random.seed(seeds[i_episode])
        random.seed(seeds[i_episode])
        env.seed(int(seeds[i_episode]))
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            # print(env.action_space) #1, 4
            # print(env.observation_space) #3, 24
            # action = [[[-0.9998976,  -0.01430972, 0.83109677]
            action = (random_state.rand(env.action_space.shape[0]) - 0.5) * 2 * env.action_space.high
            print(action)
            # print(action)
            new_state, reward, done, info = env.step(action)
            # print(new_state, reward, done, info)
            
            total_reward += reward
            # print(total_reward)
            if done:
                print(env)
                break
        print(total_reward)
        rewards.append(total_reward)
    print("Avg reward: {:0.2f}".format(np.mean(rewards)))
    print("Min reward: {:0.2f}".format(np.min(rewards)))
    print("Max reward: {:0.2f}".format(np.max(rewards)))
    print()
    env.close()

# https://minpy.readthedocs.io/en/latest/tutorial/rl_policy_gradient_tutorial/rl_policy_gradient.html