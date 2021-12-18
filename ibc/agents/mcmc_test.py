# coding=utf-8
# Copyright 2021 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ibc.ibc.mcmc."""
import time

from ibc.environments.particle import particle
from ibc.ibc.agents import mcmc
from ibc.networks import mlp_ebm
import numpy as np
import tensorflow.compat.v2 as tf

from tf_agents.environments import suite_gym
from tf_agents.train.utils import spec_utils
from ibc.networks.layers.mlp_dropout import MLPDropoutLayer
from ibc.ibc.utils import debug


class McmcTest(tf.test.TestCase):

  def _test_batched_categorical_bincount_shapes(self, batch_size):
    num_samples = 256
    probs = np.random.randn(batch_size, num_samples)
    count = 128
    for i in range(batch_size):
      probs[i] = probs[i] / probs[i].sum()
    indices_counts = mcmc.categorical_bincount(count, tf.math.log(probs),
                                               num_samples)
    assert indices_counts.shape[0] == batch_size
    assert indices_counts.shape[1] == num_samples

  def test_batched_categorical_bincount_shapes(self):
    for batch_size in [1, 2]:
      self._test_batched_categorical_bincount_shapes(batch_size)

  def test_batched_categorical_bincount_correct(self):
    eps = 1e-6
    probs = np.array([[eps, eps, 1.-eps],
                      [eps, 1.-eps, eps]])
    count = 5
    indices_counts = mcmc.categorical_bincount(count, tf.math.log(probs),
                                               probs.shape[1]).numpy()
    # Assert sampled "count" times:
    for i in range(2):
      assert indices_counts[i].sum() == count
    # Assert the most-counted probs are correct.
    assert np.argmax(indices_counts[0]) == 2
    assert np.argmax(indices_counts[1]) == 1

  def _get_network_and_time_step(self):
    env = particle.ParticleEnv()
    env = suite_gym.wrap_env(env)
    obs_spec, act_spec, _ = spec_utils.get_tensor_specs(env)
    energy_network = mlp_ebm.MLPEBM((obs_spec, act_spec),
                                    tf.TensorSpec([1]))
    return energy_network, env.reset()

  def _get_mock_energy_network(self):
    class EnergyNet(tf.keras.Model):

      def __init__(self, energy_scale=1e2):
        super(EnergyNet, self).__init__()
        self.mean = np.array([0.3, 0.4])
        self.energy_scale = energy_scale

      def call(self, x, step_type=(), network_state=(), training=()):
        """Mock network."""
        _, actions = x
        return -(tf.linalg.norm(actions - self.mean, axis=1)
                 * self.energy_scale)**2, ()

    return EnergyNet()

  def test_shapes_iterative_dfo(self):
    energy_network, time_step = self._get_network_and_time_step()
    batch_size = 2
    num_action_samples = 2048

    obs = time_step.observation
    # "Batch" the observations by replicating
    for key in obs.keys():
      batch_obs = obs[key][None, Ellipsis]
      obs[key] = tf.concat([batch_obs] * (batch_size * num_action_samples),
                           axis=0)

    init_action_samples = np.random.rand(batch_size * num_action_samples,
                                         2).astype(np.float32)
    # Forces network to create variables.
    energy_network((obs, init_action_samples), training=False)

    probs, action_samples, _ = mcmc.iterative_dfo(
        energy_network,
        batch_size,
        obs,
        init_action_samples,
        policy_state=(),
        temperature=1.0,
        num_action_samples=num_action_samples,
        min_actions=np.array([0., 0.]).astype(np.float32),
        max_actions=np.array([1., 1.]).astype(np.float32),
        num_iterations=3,
        iteration_std=1e-1,
        training=False,
        tfa_step_type=())
    assert action_samples.shape == init_action_samples.shape
    assert probs.shape[0] == batch_size * num_action_samples

  def test_correct_iterative_dfo(self):

    energy_network = self._get_mock_energy_network()
    # Forces network to create variables.
    energy_network(((), np.random.randn(1, 2).astype(np.float32)))

    batch_size = 2
    num_action_samples = 2048
    obs = ()
    init_action_samples = np.random.rand(batch_size * num_action_samples,
                                         2).astype(np.float32)

    probs, action_samples, _ = mcmc.iterative_dfo(
        energy_network,
        batch_size,
        obs,
        init_action_samples,
        policy_state=(),
        temperature=1.0,
        num_action_samples=num_action_samples,
        min_actions=np.array([0., 0.]).astype(np.float32),
        max_actions=np.array([1., 1.]).astype(np.float32),
        num_iterations=10,
        iteration_std=1e-1,
        training=False,
        tfa_step_type=())
    assert tf.linalg.norm(action_samples[np.argmax(probs)] - \
                          energy_network.mean) < 0.01

  def test_correct_langevin(self):

    energy_network = self._get_mock_energy_network()
    # Forces network to create variables.
    energy_network(((), np.random.randn(1, 2).astype(np.float32)))

    batch_size = 2
    num_action_samples = 128
    obs = ()
    init_action_samples = np.random.rand(batch_size * num_action_samples,
                                         2).astype(np.float32)

    action_samples = mcmc.langevin_actions_given_obs(
        energy_network,
        obs,
        init_action_samples,
        policy_state=(),
        min_actions=np.array([0., 0.]).astype(np.float32),
        max_actions=np.array([1., 1.]).astype(np.float32),
        training=False,
        num_iterations=25,
        num_action_samples=num_action_samples,
        tfa_step_type=())
    assert tf.linalg.norm(action_samples[0] - \
                          energy_network.mean) < 0.1

  @debug.iex
  def test_profile(self):

    class EnergyNet(tf.keras.Model):

      def __init__(self, mlp):
        super(EnergyNet, self).__init__()
        self.mlp = mlp

      def call(self, inputs, training, step_type=(), network_state=()):
        """Mock network."""
        _, actions = inputs
        return self.mlp(actions), ()

    dense = tf.keras.layers.Dense
    mlp = MLPDropoutLayer(
        [512, 512],
        rate=0,
        kernel_initializer='normal',
        bias_initializer='normal',
        dense=dense,
    )
    energy_network = EnergyNet(mlp)
    # Create stuff...
    energy_network(((), np.random.randn(1, 2).astype(np.float32)))

    batch_size = 32
    num_action_samples = 8

    obs = ()
    init_action_samples = np.random.rand(batch_size * num_action_samples,
                                         2).astype(np.float32)

    print()

    count = 100
    num_iter = 100

    t_start = None
    for _ in range(count + 1):
      action_samples = mcmc.langevin_actions_given_obs(
          energy_network,
          obs,
          init_action_samples,
          policy_state=(),
          min_actions=np.array([0., 0.]).astype(np.float32),
          max_actions=np.array([1., 1.]).astype(np.float32),
          training=False,
          num_iterations=num_iter,
          num_action_samples=num_action_samples,
          tfa_step_type=())
      if t_start is None:
        t_start = time.time()
    dt = time.time() - t_start
    print(dt / (count * num_iter))
    print("\n\n")


if __name__ == '__main__':
  tf.test.main()
