import json
import logging
import os
import random
import sys
import numpy as np
from collections import deque
from os import path

import gym
from gym import wrappers

# import tensorflow as tf
import tensorflow.compat.v1 as tf

from misio.util import generate_deterministic_seeds
from tqdm import trange

logging.getLogger('tensorflow').disabled = True

tf.disable_v2_behavior()
# tf.logging.set_verbosity(tf.logging.ERROR)


class ActorCritic(object):
    def __init__(self, env_to_use):
        # init hyperparamas
        self.n_test_games = 100
        self.dropout_actor = 0.5			# dropout rate for actor (0 = no dropout)
        self.dropout_critic = 0.5			# dropout rate for critic (0 = no dropout)
        self.num_episodes = 100		# number of episodes, default: 15 000
        # default 10000 !!!! max number of steps per episode (unless env has a lower hardcoded limit)
        self.max_steps_ep = 1000
        self.tau = 1e-2				# soft target update rate
        # capacity of experience replay memory
        self.replay_memory_capacity = int(1e5)
        # size of minibatch from experience replay memory for updates
        self.minibatch_size = 1024
        # scale of the exploration noise process (1.0 is the range of each action dimension)
        self.initial_noise_scale = 0.1
        # decay rate (per episode) of the scale of the exploration noise process
        self.noise_decay = 0.99
        # mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
        self.exploration_mu = 0.0
        # theta parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
        self.exploration_theta = 0.15
        # sigma parameter for the exploration noise process: dXt = theta*(mu-Xt	)*dt + sigma*dWt
        self.exploration_sigma = 0.2

        # init environment
        self.PATH = 'tmp/'
        self.env_to_use = env_to_use
        self.env = gym.make(env_to_use)
        # Get total number of dimensions in state
        self.state_dim = np.prod(np.array(self.env.observation_space.shape))
        # Assuming continuous action space
        self.action_dim = np.prod(np.array(self.env.action_space.shape))

        self.info = {}
        np.set_printoptions(threshold=0)
        # used for O(1) popleft() operation
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)
        tf.reset_default_graph()

        # TENSORFLOW STUFF
        # placeholders
        self.state_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.state_dim])
        self.action_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.action_dim])
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.state_dim])
        # indicators (go into target computation)
        self.is_not_terminal_ph = tf.placeholder(
            dtype=tf.float32, shape=[None])
        self.is_training_ph = tf.placeholder(
            dtype=tf.bool, shape=())  # for dropout

        # episode counter
        self.episodes = tf.Variable(0.0, trainable=False, name='episodes')
        self.episode_inc_op = self.episodes.assign_add(1)

    def add_to_memory(self, experience):
        self.replay_memory.append(experience)

    def sample_from_memory(self, minibatch_size):
        return random.sample(self.replay_memory, minibatch_size)

    def generate_actor_network(self, s, trainable, reuse):
        ''' ACTOR NETWORK '''
        hidden = tf.layers.dense(
            s, 8, activation=tf.nn.relu, trainable=trainable, name='dense', reuse=reuse)
        hidden_drop = tf.layers.dropout(
            hidden, rate=self.dropout_actor, training=trainable & self.is_training_ph)
        hidden_2 = tf.layers.dense(
            hidden_drop, 8, activation=tf.nn.relu, trainable=trainable, name='dense_1', reuse=reuse)
        hidden_drop_2 = tf.layers.dropout(
            hidden_2, rate=self.dropout_actor, training=trainable & self.is_training_ph)
        hidden_3 = tf.layers.dense(
            hidden_drop_2, 8, activation=tf.nn.relu, trainable=trainable, name='dense_2', reuse=reuse)
        hidden_drop_3 = tf.layers.dropout(
            hidden_3, rate=self.dropout_actor, training=trainable & self.is_training_ph)
        actions_unscaled = tf.layers.dense(
            hidden_drop_3, self.action_dim, trainable=trainable, name='dense_3', reuse=reuse)
        actions = self.env.action_space.low + tf.nn.sigmoid(actions_unscaled)*(
            self.env.action_space.high - self.env.action_space.low)  # bound the actions to the valid range
        return actions

    def generate_critic_network(self, s, a, trainable, reuse):
        ''' CRITIC NETWORK '''
        state_action = tf.concat([s, a], axis=1)
        hidden = tf.layers.dense(
            state_action, 8, activation=tf.nn.relu, trainable=trainable, name='dense', reuse=reuse)
        hidden_drop = tf.layers.dropout(
            hidden, rate=self.dropout_critic, training=trainable & self.is_training_ph)
        hidden_2 = tf.layers.dense(
            hidden_drop, 8, activation=tf.nn.relu, trainable=trainable, name='dense_1', reuse=reuse)
        hidden_drop_2 = tf.layers.dropout(
            hidden_2, rate=self.dropout_critic, training=trainable & self.is_training_ph)
        hidden_3 = tf.layers.dense(
            hidden_drop_2, 8, activation=tf.nn.relu, trainable=trainable, name='dense_2', reuse=reuse)
        hidden_drop_3 = tf.layers.dropout(
            hidden_3, rate=self.dropout_critic, training=trainable & self.is_training_ph)
        q_values = tf.layers.dense(
            hidden_drop_3, 1, trainable=trainable, name='dense_3', reuse=reuse)
        return q_values

    def initSession(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print("Model init.")
        return sess, saver

    def restoreSession(self):
        self.env.close()
        self.env = gym.make(self.env_to_use)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        filepath = self.PATH + 'model.ckpt'
        saver.restore(sess, filepath)
        print("Model restored.")
        return sess, saver

    def makeSteps(self):

        # actor network
        with tf.variable_scope('actor'):
            # Policy's outputted action for each state_ph (for generating actions and training the critic)
            self.actions = self.generate_actor_network(
                self.state_ph, trainable=True, reuse=False)

        # slow target actor network
        with tf.variable_scope('slow_target_actor', reuse=False):
            # Slow target policy's outputted action for each next_state_ph (for training the critic)
            # use stop_gradient to treat the output values as constant targets when doing backprop
            slow_target_next_actions = tf.stop_gradient(
                self.generate_actor_network(self.next_state_ph, trainable=False, reuse=False))

        with tf.variable_scope('critic') as scope:
            # Critic applied to state_ph and a given action (for training critic)
            q_values_of_given_actions = self.generate_critic_network(
                self.state_ph, self.action_ph, trainable=True, reuse=False)
            # Critic applied to state_ph and the current policy's outputted actions for state_ph (for training actor via deterministic policy gradient)
            q_values_of_suggested_actions = self.generate_critic_network(
                self.state_ph, self.actions, trainable=True, reuse=True)

        # slow target critic network
        with tf.variable_scope('slow_target_critic', reuse=False):
            # Slow target critic applied to slow target actor's outputted actions for next_state_ph (for training critic)
            slow_q_values_next = tf.stop_gradient(self.generate_critic_network(
                self.next_state_ph, slow_target_next_actions, trainable=False, reuse=False))

        # isolate vars for each network
        actor_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        slow_target_actor_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor')
        critic_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        slow_target_critic_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_critic')

        # update values for slowly-changing targets towards current actor and critic
        update_slow_target_ops = []
        for i, slow_target_actor_var in enumerate(slow_target_actor_vars):
            update_slow_target_actor_op = slow_target_actor_var.assign(
                self.tau*actor_vars[i]+(1-self.tau)*slow_target_actor_var)
            update_slow_target_ops.append(update_slow_target_actor_op)

        for i, slow_target_var in enumerate(slow_target_critic_vars):
            update_slow_target_critic_op = slow_target_var.assign(
                self.tau*critic_vars[i]+(1-self.tau)*slow_target_var)
            update_slow_target_ops.append(update_slow_target_critic_op)

        update_slow_targets_op = tf.group(
            *update_slow_target_ops, name='update_slow_targets')

        # ONE STEP
        # TD targets y_i for (s,a) from experience replay
        gamma = 0.99
        l2_reg_actor = 1e-6			# L2 regularization factor for the actor
        l2_reg_critic = 1e-6		# L2 regularization factor for the critic

        targets = tf.expand_dims(self.reward_ph, 1) + tf.expand_dims(
            self.is_not_terminal_ph, 1) * gamma * slow_q_values_next
        # 1-step temporal difference errors
        td_errors = targets - q_values_of_given_actions
        # critic loss function (mean-square value error with regularization)
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        for var in critic_vars:
            if not 'bias' in var.name:
                critic_loss += l2_reg_critic * 0.5 * tf.nn.l2_loss(var)

        # OPTIMIZERS
        lr_critic = 1e-3
        lr_actor = 1e-3
        # critic optimizers
        critic_train_op = tf.train.AdamOptimizer(
            lr_critic**self.episodes).minimize(critic_loss)
        # actor loss function (mean Q-values under current policy with regularization)
        actor_loss = -1*tf.reduce_mean(q_values_of_suggested_actions)
        for var in actor_vars:
            if not 'bias' in var.name:
                actor_loss += l2_reg_actor * 0.5 * tf.nn.l2_loss(var)
        # actor optimizer
        # the gradient of the mean Q-values wrt actor params is the deterministic policy gradient (keeping critic params fixed)
        actor_train_op = tf.train.AdamOptimizer(
            lr_actor**self.episodes).minimize(actor_loss, var_list=actor_vars)

        sess, saver = self.initSession()

        total_steps = 0
        for ep in range(self.num_episodes):

            total_reward = 0
            steps_in_ep = 0

            # Initialize exploration noise process
            self.noise_process = np.zeros(self.action_dim)
            self.noise_scale = (self.initial_noise_scale * self.noise_decay**ep) * \
                (self.env.action_space.high - self.env.action_space.low)

            # Initial state
            self.observation = self.env.reset()
            for t in range(self.max_steps_ep):

                # choose action based on deterministic policy
                action_for_state, = sess.run(self.actions,
                                             feed_dict={self.state_ph: self.observation[None], self.is_training_ph: False})

                # add temporally-correlated exploration noise to action (using an Ornstein-Uhlenbeck process)
                self.noise_process = self.exploration_theta * \
                    (self.exploration_mu - self.noise_process) + \
                    self.exploration_sigma*np.random.randn(self.action_dim)
                action_for_state += self.noise_scale*self.noise_process

                # take step
                next_observation, reward, done, _info = self.env.step(
                    action_for_state)
                total_reward += reward

                self.add_to_memory((self.observation, action_for_state, reward, next_observation,
                                    # is next_observation a terminal state?
                                    # 0.0 if done and not env.env._past_limit() else 1.0))
                                    0.0 if done else 1.0))

                # update network weights to fit a minibatch of experience
                if total_steps % 1 == 0 and len(self.replay_memory) >= self.minibatch_size:

                    # grab N (s,a,r,s') tuples from replay memory
                    minibatch = self.sample_from_memory(self.minibatch_size)

                    # update the critic and actor params using mean-square value error and deterministic policy gradient, respectively
                    _, _ = sess.run([critic_train_op, actor_train_op],
                                    feed_dict={
                                    self.state_ph: np.asarray([elem[0] for elem in minibatch]),
                                    self.action_ph: np.asarray([elem[1] for elem in minibatch]),
                                    self.reward_ph: np.asarray([elem[2] for elem in minibatch]),
                                    self.next_state_ph: np.asarray([elem[3] for elem in minibatch]),
                                    self.is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch]),
                                    self.is_training_ph: True})

                    # update slow actor and critic targets towards current actor and critic
                    _ = sess.run(update_slow_targets_op)

                self.observation = next_observation
                total_steps += 1
                steps_in_ep += 1

                if done:
                    # Increment episode counter
                    _ = sess.run(self.episode_inc_op)
                    break

            print('Episode %2i, Reward: %7.3f, Steps: %i' %
                  (ep, total_reward, steps_in_ep))
        filepath = self.PATH + 'model.ckpt'
        save_path = saver.save(sess, filepath)
        print("Model saved in path:", save_path)

    def printResults(self, rewards):
        print("Avg reward: {:0.2f}".format(np.mean(rewards)))
        print("Min reward: {:0.2f}".format(np.min(rewards)))
        print("Max reward: {:0.2f}".format(np.max(rewards)))
        print()

    def testingLoop(self):
        self.makeSteps()
        sess, saver = self.restoreSession()

        rewards = []
        for i_episode in trange(self.n_test_games):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                self.env.render()

                action_for_state, = sess.run(self.actions, feed_dict={
                                             self.state_ph: self.observation[None], self.is_training_ph: False})
                self.noise_process = self.exploration_theta * \
                    (self.exploration_mu - self.noise_process) + \
                    self.exploration_sigma*np.random.randn(self.action_dim)
                action_for_state += self.noise_scale*self.noise_process
                new_state, reward, done, info = self.env.step(action_for_state)
                self.observation = new_state

                total_reward += reward

                if done:
                    break

            rewards.append(total_reward)

        self.printResults(rewards)
        env.close()


# main
if __name__ == "__main__":
    envir = "BipedalWalker-v3"
	# envir = "Pendulum-v0"
    ac = ActorCritic(envir)
    ac.testingLoop()

