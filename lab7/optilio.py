#!/usr/bin/env python3
import json
import os
import random
import sys
from collections import deque
from os import path

import gym
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from gym import wrappers
from misio.util import generate_deterministic_seeds
from tqdm import trange
from run_trained_model import tfLoad
tf.disable_v2_behavior()


def pednulum():
    actions, state_ph, is_training_ph = tfLoad("Pendulum-v0")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "tmpPendulum/model.ckpt")

    num_games = int(input())
    for _ in range(num_games):
        while True:
            state = input()
            if state == "END":
                break
            observation = np.array(
                [float(i) for i in state[1:-1].split(' ') if len(i) > 0])
            action_for_state, = sess.run(
                actions, feed_dict={state_ph: observation[None], is_training_ph: False})
            print(action_for_state, flush=True)


def bipedalWalker():
    actions, state_ph, is_training_ph = tfLoad("BipedalWalker-v3")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "tmpBipedal/model.ckpt")

    num_games = int(input())
    for _ in range(num_games):
        while True:
            state = input()
            if state == "END":
                break
            observation = np.array(
                [float(i) for i in state[1:-1].split(' ') if len(i) > 0])
            action_for_state, = sess.run(
                actions, feed_dict={state_ph: observation[None], is_training_ph: False})
            print(action_for_state, flush=True)


if __name__ == "__main__":
    game = input()
    if game == "BipedalWalker-v3":
        bipedalWalker()
    elif game == "Pendulum-v0":
        pednulum()
    else:
        raise ValueError("Unexpected game: {}".format(game))
