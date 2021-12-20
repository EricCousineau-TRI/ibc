import time

import numpy as np
import tensorflow.compat.v2 as tf

from ibc.ibc.utils import debug


def make_mlp(hidden_sizes):
  net = tf.keras.Sequential()
  for hidden_size in hidden_sizes:
    net.add(tf.keras.layers.Dense(hidden_size))
  net.add(tf.keras.layers.Dense(1))
  return net


# @tf.function
def gradient(f, x):
  with tf.GradientTape() as g:
    g.watch(x)
    f_value = f(x, training=False)
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

  net = make_mlp([256, 256])
  net(yhs)
  print(net.summary())
  # net = tf.function(net)

  count = 10
  num_iter = 100

  t_start = None
  for _ in range(count + 1):

    out = fake_langevin(net, yhs, num_iter)
    out = out.numpy()

    if t_start is None:
      t_start = time.time()

  dt = time.time() - t_start
  print(dt / count)
  print("\n\n")


@debug.iex
def main():
  tf.config.run_functions_eagerly(False)
  test_profile()


if __name__ == "__main__":
  main()
