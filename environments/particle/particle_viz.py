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

"""Visualization methods for the particle envs."""

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import einops

from tf_agents.utils import nest_utils

def make_vector(step):
  """step is either an obs or an act."""
  return np.hstack(list(step.values()))


def make_vector_traj(log):
  """log is either an obs_log or an act_log."""
  vector_traj = []
  for step in log:
    vector_traj.append(make_vector(step))
  return np.array(vector_traj)


def plot_particle_2d_ebm(
    info, obs, *, grid_size, alpha=None, temperature=10.0
):
  action_x = tf.linspace(0.0, 1.0, num=grid_size)
  action_y = tf.linspace(0.0, 1.0, num=grid_size)
  action_y_grid, action_x_grid = tf.meshgrid(action_x, action_y)

  action_xs = einops.rearrange(action_x_grid, "H W -> (H W)")
  action_ys = einops.rearrange(action_y_grid, "H W -> (H W)")
  ys = einops.rearrange([action_xs, action_ys], "C N -> N C")
  N = ys.shape[0]

  def reshape_obs(x):
    return einops.repeat(x, "D -> N 1 D", N=N)

  obs_norm, _ = info.obs_norm(obs)
  obs_norm = tf.nest.map_structure(reshape_obs, obs_norm)
  yhs = info.act_norm(ys)

  Z_grid, _ = info.net((obs_norm, yhs))

  print(f"min energy: {tf.reduce_min(Z_grid)}")
  print(f"max energy: {tf.reduce_max(Z_grid)}")
  print()

  # Show probabilities used for sampling.
  Z_grid = tf.nn.softmax(Z_grid / temperature, axis=0)
  Z_grid, _ = tf.linalg.normalize(Z_grid, axis=0)
  Z_grid = einops.rearrange(Z_grid, "(H W) -> H W", H=grid_size)
  mesh = plt.pcolormesh(
    action_x_grid.numpy(),
    action_y_grid.numpy(),
    Z_grid.numpy(),
    cmap="cool",
    alpha=alpha,
    shading="auto",
  )
  return mesh


def visualize_2d(
    obs_log, act_log, *, net_info=None, ax=None, fig=None, show=False, last_big=False):
  """If the environment is 2d, render a top-down image."""

  # Assert it's 2D
  assert len(obs_log[0]['pos_agent']) == 2

  if ax is None:
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)

  if net_info is not None:
    plot_particle_2d_ebm(net_info, obs_log[-1], grid_size=50)

  # Since when render is called we don't know what the actions will be,
  # We may need to ignore the last obs.
  if len(obs_log) != len(act_log):
    if len(obs_log) == len(act_log) + 1:
      obs_log_ = obs_log[:-1]
    else:
      raise ValueError('Wrong length logs.')
  else:
    obs_log_ = obs_log

  # Visualize observations.
  pos_first_goal = obs_log[0]['pos_first_goal']
  ax.add_patch(plt.Circle(
      (pos_first_goal[0], pos_first_goal[1]), 0.01, color='g'))
  pos_second_goal = obs_log[0]['pos_second_goal']
  ax.add_patch(plt.Circle(
      (pos_second_goal[0], pos_second_goal[1]), 0.01, color='b'))

  # Now obs_log_ might be empty, in which case return.
  if not obs_log_:
    return fig, ax

  # Visualize actions.
  act_traj = make_vector_traj(act_log)
  ax.scatter(
      act_traj[:, 0], act_traj[:, 1], marker='x', s=100, alpha=0.1, color='red')

  for i in range(len(obs_log_)-1):
    alpha = float(i)/len(obs_log_)
    pos_agent_k = obs_log_[i]['pos_agent']
    pos_agent_kplus1 = obs_log_[i+1]['pos_agent']
    pos_agent_2step = np.stack((pos_agent_k, pos_agent_kplus1))
    ax.plot(pos_agent_2step[:, 0], pos_agent_2step[:, 1],
            alpha=alpha, linestyle=':', color='black')
  if last_big:
    ax.scatter(obs_log_[-1]['pos_agent'][0], obs_log_[-1]['pos_agent'][1],
               marker='o', s=50, color='black')
    ax.scatter(act_traj[-1, 0], act_traj[-1, 1], marker='x', color='red', s=100)
  if show:
    plt.show()
  return fig, ax


def visualize_nd(obs_log, act_log, axes=None, fig=None, show=True,
                 xlim=None, ylim=None, last_big=False):
  """For any dimension, visualize signals over time."""
  if len(obs_log) != len(act_log):
    if len(obs_log) == len(act_log) + 1:
      obs_log_ = obs_log[:-1]
    else:
      raise ValueError('Wrong length logs.')
  else:
    obs_log_ = obs_log
  dims = obs_log[0]['pos_agent'].shape[0]
  if axes is None:
    fig, axes = plt.subplots(dims, figsize=(12, 2*dims))
    if dims == 1:
      axes = [axes]
  if xlim is not None:
    _ = [ax.set_xlim(xlim) for ax in axes]
  if ylim is not None:
    _ = [ax.set_ylim(ylim) for ax in axes]

  _ = [ax.set(xlabel='time', ylabel='position') for ax in axes]
  if not act_log:
    return fig, axes

  obs_traj = make_vector_traj(obs_log_)
  act_traj = make_vector_traj(act_log)
  for i in range(dims):
    axes[i].plot(
        np.arange(len(obs_log_)), obs_traj[:, i], 'black', label='agent')
    axes[i].plot(
        np.arange(len(obs_log_)),
        obs_traj[:, i + 2 * dims],
        'g',
        label='1st_point')
    axes[i].plot(
        np.arange(len(obs_log_)),
        obs_traj[:, i + 3 * dims],
        'b',
        label='2nd_point')
    axes[i].scatter(
        np.arange(len(obs_log_)),
        act_traj[:, i],
        marker='x',
        s=100,
        alpha=0.1,
        label='agent_goal',
        color='red')
    if last_big:
      axes[i].scatter(len(obs_log_), obs_traj[-1, i],
                      marker='o', s=50, color='black')
      axes[i].scatter(len(obs_log_), obs_traj[-1, i+2*dims],
                      marker='o', s=50, color='g')
      axes[i].scatter(len(obs_log_), obs_traj[-1, i+3*dims],
                      marker='o', s=50, color='b')
      axes[i].scatter(len(obs_log_), act_traj[-1, i],
                      marker='x', s=100, color='red')
  if show:
    plt.legend()
    plt.show()
  return fig, axes
