# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import math
from actor_network import ActorNetwork
import tensorflow as tf


class ThreelinkArm(gym.Env):
    """This class creates a continous-state maze problem given a map."""

    metadata = {
        'render.modes': ['robot', 'joint'],
    }

    def __init__(self):
        self._init_setup()
        self.viewer = None
        self.action_space = spaces.Box(self.act_low,self.act_high)
        self.observation_space = spaces.Box(self.obs_low,self.obs_high)
        self._seed()
        self._reset()
        self.dt = 0.01
        self.sess = tf.InteractiveSession()
        self.actor_network =  ActorNetwork(self.sess,self.observation_space.shape[0],self.action_space.shape[0])
        self.goal_state = np.zeros(shape=3)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action, type(action))
        state = self.state + self.dt*action

        if (np.random.uniform(0., 1.) > self.motion_noise):
            # Motion length is a truncated normal random variable.
            self.state[0] = state[0]
            self.state[1] = state[1]
            self.state[2] = state[2]

        done = self._is_goal(self.state)
        reward = self._compute_reward(self.state, action, self.actor_network)
        return self.state, reward, done, {}

    def _reset(self):
        self.initial_state = np.add(np.array([0., 0., 0.]), np.random.rand(3))
        self.goal_state = np.add(np.array([1., 1.,1.]), np.random.rand(3))
        self.state = np.copy(self.initial_state)
        return self.state

    def _init_setup(self):
        self.obs_high = np.array([3.14, 3.14, 3.14])
        self.obs_low = np.array([-3.14, -3.14, -3.14])
        self.act_high = np.array([10, 10, 10])
        self.act_low = np.array([-10, -10, -10])

        self.motion_noise = 0.05  # probability of no-motion (staying in same state)

    def _compute_reward(self, state, action, actor_network):
        # cost = ||(Y dot u_ns/u_ns dot u_ns)*u_ns- u_ns||^2
        u_ns = actor_network.action(state)
        tmp = (action.dot(u_ns)/u_ns.dot(u_ns))*u_ns - u_ns
        reward = -tmp.dot(tmp)
        return reward

    def _is_goal(self, state):
        tmp = math.sqrt((state[0] - self.goal_state[0])**2)
        return tmp < 0.01

    def _render(self, mode='human', close=False):
        pass

    def _update_actor_network(self,actor_network):
        self.actor_network = actor_network
        return self.actor_network
