#!/usr/bin/env python3
import json
import os
import random
import sys
from collections import deque
from os import path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import gym
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from gym import wrappers
from misio.util import generate_deterministic_seeds
from tqdm import trange

tf.disable_v2_behavior()


# -----------
ENVIRONMENT = "Pendulum-v0"

def tfLoad(environment_name):
	RENDER_SOME = False
	# hyperparameters
	gamma = 0.99				# reward discount factor
	h1_actor = 8				# hidden layer 1 size for the actor
	h2_actor = 8				# hidden layer 2 size for the actor
	h3_actor = 8				# hidden layer 3 size for the actor
	h1_critic = 8				# hidden layer 1 size for the critic
	h2_critic = 8				# hidden layer 2 size for the critic
	h3_critic = 8				# hidden layer 3 size for the critic
	lr_actor = 1e-3				# learning rate for the actor
	lr_critic = 1e-3			# learning rate for the critic
	lr_decay = 1				# learning rate decay (per episode)
	l2_reg_actor = 1e-6			# L2 regularization factor for the actor
	l2_reg_critic = 1e-6		# L2 regularization factor for the critic
	dropout_actor = 0			# dropout rate for actor (0 = no dropout)
	dropout_critic = 0			# dropout rate for critic (0 = no dropout)
	num_episodes = 500		# number of episodes, default: 15 000
	max_steps_ep = 10000	# default max number of steps per episode (unless env has a lower hardcoded limit)
	tau = 1e-2				# soft target update rate
	train_every = 1			# number of steps to run the policy (and collect experience) before updating network weights
	replay_memory_capacity = int(1e5)	# capacity of experience replay memory
	minibatch_size = 1024	# size of minibatch from experience replay memory for updates
	initial_noise_scale = 0.1	# scale of the exploration noise process (1.0 is the range of each action dimension)
	noise_decay = 0.99		# decay rate (per episode) of the scale of the exploration noise process
	exploration_mu = 0.0	# mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
	exploration_theta = 0.15 # theta parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
	exploration_sigma = 0.2	# sigma parameter for the exploration noise process: dXt = theta*(mu-Xt	)*dt + sigma*dWt

	# game parameters
	env = gym.make(environment_name)
	state_dim = np.prod(np.array(env.observation_space.shape)) 	# Get total number of dimensions in state
	action_dim = np.prod(np.array(env.action_space.shape))		# Assuming continuous action space

	tf.reset_default_graph()

	# placeholders
	state_ph = tf.placeholder(dtype=tf.float32, shape=[None,state_dim])
	action_ph = tf.placeholder(dtype=tf.float32, shape=[None,action_dim])
	reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
	next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None,state_dim])
	is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # indicators (go into target computation)
	is_training_ph = tf.placeholder(dtype=tf.bool, shape=()) # for dropout

	# episode counter
	episodes = tf.Variable(0.0, trainable=False, name='episodes')
	episode_inc_op = episodes.assign_add(1)

	# will use this to initialize both the actor network its slowly-changing target network with same structure
	def generate_actor_network(s, trainable, reuse):
		hidden = tf.layers.dense(s, h1_actor, activation = tf.nn.relu, trainable = trainable, name = 'dense', reuse = reuse)
		hidden_drop = tf.layers.dropout(hidden, rate = dropout_actor, training = trainable & is_training_ph)
		hidden_2 = tf.layers.dense(hidden_drop, h2_actor, activation = tf.nn.relu, trainable = trainable, name = 'dense_1', reuse = reuse)
		hidden_drop_2 = tf.layers.dropout(hidden_2, rate = dropout_actor, training = trainable & is_training_ph)
		hidden_3 = tf.layers.dense(hidden_drop_2, h3_actor, activation = tf.nn.relu, trainable = trainable, name = 'dense_2', reuse = reuse)
		hidden_drop_3 = tf.layers.dropout(hidden_3, rate = dropout_actor, training = trainable & is_training_ph)
		actions_unscaled = tf.layers.dense(hidden_drop_3, action_dim, trainable = trainable, name = 'dense_3', reuse = reuse)
		actions = env.action_space.low + tf.nn.sigmoid(actions_unscaled)*(env.action_space.high - env.action_space.low) # bound the actions to the valid range
		return actions

	# actor network
	with tf.variable_scope('actor'):
		# Policy's outputted action for each state_ph (for generating actions and training the critic)
		actions = generate_actor_network(state_ph, trainable = True, reuse = False)

	# slow target actor network
	with tf.variable_scope('slow_target_actor', reuse=False):
		# Slow target policy's outputted action for each next_state_ph (for training the critic)
		# use stop_gradient to treat the output values as constant targets when doing backprop
		slow_target_next_actions = tf.stop_gradient(generate_actor_network(next_state_ph, trainable = False, reuse = False))

	# will use this to initialize both the critic network its slowly-changing target network with same structure
	def generate_critic_network(s, a, trainable, reuse):
		state_action = tf.concat([s, a], axis=1)
		hidden = tf.layers.dense(state_action, h1_critic, activation = tf.nn.relu, trainable = trainable, name = 'dense', reuse = reuse)
		hidden_drop = tf.layers.dropout(hidden, rate = dropout_critic, training = trainable & is_training_ph)
		hidden_2 = tf.layers.dense(hidden_drop, h2_critic, activation = tf.nn.relu, trainable = trainable, name = 'dense_1', reuse = reuse)
		hidden_drop_2 = tf.layers.dropout(hidden_2, rate = dropout_critic, training = trainable & is_training_ph)
		hidden_3 = tf.layers.dense(hidden_drop_2, h3_critic, activation = tf.nn.relu, trainable = trainable, name = 'dense_2', reuse = reuse)
		hidden_drop_3 = tf.layers.dropout(hidden_3, rate = dropout_critic, training = trainable & is_training_ph)
		q_values = tf.layers.dense(hidden_drop_3, 1, trainable = trainable, name = 'dense_3', reuse = reuse)
		return q_values

	with tf.variable_scope('critic') as scope:
		# Critic applied to state_ph and a given action (for training critic)
		q_values_of_given_actions = generate_critic_network(state_ph, action_ph, trainable = True, reuse = False)
		# Critic applied to state_ph and the current policy's outputted actions for state_ph (for training actor via deterministic policy gradient)
		q_values_of_suggested_actions = generate_critic_network(state_ph, actions, trainable = True, reuse = True)

	# slow target critic network
	with tf.variable_scope('slow_target_critic', reuse=False):
		# Slow target critic applied to slow target actor's outputted actions for next_state_ph (for training critic)
		slow_q_values_next = tf.stop_gradient(generate_critic_network(next_state_ph, slow_target_next_actions, trainable = False, reuse = False))

	# isolate vars for each network
	actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
	slow_target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor')
	critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
	slow_target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_critic')

	# update values for slowly-changing targets towards current actor and critic
	update_slow_target_ops = []
	for i, slow_target_actor_var in enumerate(slow_target_actor_vars):
		update_slow_target_actor_op = slow_target_actor_var.assign(tau*actor_vars[i]+(1-tau)*slow_target_actor_var)
		update_slow_target_ops.append(update_slow_target_actor_op)

	for i, slow_target_var in enumerate(slow_target_critic_vars):
		update_slow_target_critic_op = slow_target_var.assign(tau*critic_vars[i]+(1-tau)*slow_target_var)
		update_slow_target_ops.append(update_slow_target_critic_op)

	update_slow_targets_op = tf.group(*update_slow_target_ops, name='update_slow_targets')

	# One step TD targets y_i for (s,a) from experience replay
	# = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
	# = r_i if s' terminal
	targets = tf.expand_dims(reward_ph, 1) + tf.expand_dims(is_not_terminal_ph, 1) * gamma * slow_q_values_next

	# 1-step temporal difference errors
	td_errors = targets - q_values_of_given_actions

	# critic loss function (mean-square value error with regularization)
	critic_loss = tf.reduce_mean(tf.square(td_errors))
	for var in critic_vars:
		if not 'bias' in var.name:
			critic_loss += l2_reg_critic * 0.5 * tf.nn.l2_loss(var)

	# critic optimizer
	critic_train_op = tf.train.AdamOptimizer(lr_critic*lr_decay**episodes).minimize(critic_loss)

	# actor loss function (mean Q-values under current policy with regularization)
	actor_loss = -1*tf.reduce_mean(q_values_of_suggested_actions)
	for var in actor_vars:
		if not 'bias' in var.name:
			actor_loss += l2_reg_actor * 0.5 * tf.nn.l2_loss(var)

	# actor optimizer
	# the gradient of the mean Q-values wrt actor params is the deterministic policy gradient (keeping critic params fixed)
	actor_train_op = tf.train.AdamOptimizer(lr_actor*lr_decay**episodes).minimize(actor_loss, var_list=actor_vars)

	return actions, state_ph, is_training_ph


if __name__ == "__main__":
	actions, state_ph, is_training_ph = tfLoad(ENVIRONMENT)
	n_games = 10
	env = gym.make(ENVIRONMENT)

	
	sess = tf.Session()	
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	# Restore variables from disk.
	saver.restore(sess, "tmp/model.ckpt")
	print("Model restored.")


	rewards = []
	for i_episode in trange(n_games):
		observation = env.reset()
		done = False
		total_reward = 0
		while not done:	
			env.render()
			action_for_state, = sess.run(actions, feed_dict = {state_ph: observation[None], is_training_ph: False})
			# print(action_for_state, flush=True)

			new_state, reward, done, info = env.step(action_for_state)
			observation = new_state

			total_reward += reward

			if done:
				# print(env)
				break
			rewards.append(total_reward)

	print("Avg reward: {:0.2f}".format(np.mean(rewards)))
	print("Min reward: {:0.2f}".format(np.min(rewards)))
	print("Max reward: {:0.2f}".format(np.max(rewards)))
	env.close()