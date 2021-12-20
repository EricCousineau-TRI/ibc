import time

import numpy as np
import tensorflow.compat.v2 as tf

from ibc.ibc.agents import mcmc
from ibc.ibc.utils import debug
from ibc.networks.layers.mlp_dropout import MLPDropoutLayer


class EnergyNet(tf.keras.Model):
  def __init__(self, mlp):
    super(EnergyNet, self).__init__()
    self.mlp = mlp

  def call(self, inputs, training, step_type=(), network_state=()):
    """Mock network."""
    _, actions = inputs
    return self.mlp(actions), ()


@tf.function
def fake_langevin(energy_network, obs, init_action_samples, num_iter):
  out = None
  for _ in range(num_iter):
    de_dact = mcmc.gradient_wrt_act(
      energy_network,
      obs,
      init_action_samples,
      training=False,
      network_state=(),
      tfa_step_type=(),
      apply_exp=False)
    if out is None:
      out = de_dact
    else:
      out += de_dact
  return out


def test_profile():

  dense = tf.keras.layers.Dense
  mlp = MLPDropoutLayer(
      [512, 512, 1],
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

  count = 10
  num_iter = 100

  t_start = None
  for _ in range(count + 1):
    out = fake_langevin(energy_network, obs, init_action_samples, num_iter)
    # action_samples = mcmc.langevin_actions_given_obs(
    #     energy_network,
    #     obs,
    #     init_action_samples,
    #     policy_state=(),
    #     min_actions=np.array([0., 0.]).astype(np.float32),
    #     max_actions=np.array([1., 1.]).astype(np.float32),
    #     training=False,
    #     num_iterations=num_iter,
    #     num_action_samples=num_action_samples,
    #     tfa_step_type=())
    if t_start is None:
      t_start = time.time()
  dt = time.time() - t_start
  print(dt / count)
  print("\n\n")


@debug.iex
def main():
  test_profile()


if __name__ == "__main__":
  main()
