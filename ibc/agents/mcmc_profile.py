import time

import numpy as np
import tensorflow.compat.v2 as tf

from ibc.ibc.agents import mcmc
from ibc.ibc.utils import debug


class SimpleMlp(tf.keras.Model):
  def __init__(self, hidden_sizes):
    super().__init__()

    self._fc_layers = []
    for l in range(len(hidden_sizes)):
      self._fc_layers.append(tf.keras.layers.Dense(hidden_sizes[l]))
    self._final = tf.keras.layers.Dense(1)

  def _mlp(self, x, training):
    for l in range(len(self._fc_layers)):
      x = self._fc_layers[l](x, training=training)
    x = self._final(x, training=training)
    return x

  def call(self, x, training, step_type=(), network_state=()):
    """Mock network."""
    return self._mlp(x, training), ()


def gradient(f, x):
  with tf.GradientTape() as g:
    g.watch(x)
    f_value, _ = f(x, training=False)
  df_dx = g.gradient(f_value, x)
  return df_dx


@tf.function
def fake_langevin(net, yhs, num_iter):
  out = tf.zeros_like(yhs)
  for _ in range(num_iter):
    de_dact = gradient(net, yhs)
    out += de_dact
  return out


def test_profile():
  N = 32
  L = 8
  DimY = 2

  yhs = np.random.rand(N * L, DimY).astype(np.float32)
  yhs = tf.convert_to_tensor(yhs)

  net = SimpleMlp([256, 256])
  net(yhs)
  print(net.summary())

  count = 10
  num_iter = 100

  t_start = None
  for _ in range(count + 1):

    out = fake_langevin(net, yhs, num_iter)

    if t_start is None:
      t_start = time.time()

  dt = time.time() - t_start
  print(dt / count)
  print("\n\n")


@debug.iex
def main():
  # tf.config.run_functions_eagerly(True)
  test_profile()


if __name__ == "__main__":
  main()
