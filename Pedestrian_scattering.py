# IMPORTS
import matplotlib.pyplot as plt

import numpy as onp
import jax.numpy as np
from jax import random
from jax import jit
from jax import vmap
from jax import lax
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_debug_nans', True)
# config.update('jax_traceback_filtering', 'off')

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

from collections import namedtuple

vectorize = np.vectorize

from functools import partial
from simulator.utils import normal, goal_velocity_force, _ttc_force_tot
from simulator.force import ttc_force_tot, goal_velocity_force_tot, ttc_visual_force_tot
from simulator.vision import bearing_angle_tot, identity_visual_interaction
from simulator.render import render
from simulator.dynamics import pedestrian, PedestrianState, StraightWall

# YOUR CHOICE!
DIST = 2.

N = 2
dt = 0.001
delta = 20
time_step = dt * delta
frame_size = 1.5 + DIST + 1.5
key = random.PRNGKey(0)
# displacement, shift = space.periodic(box_size)
displacement, shift = space.free()

V_key, pos_key = random.split(key)

def force_fn(state):
    body_force = quantity.force(energy.soft_sphere_pair(displacement, sigma=2.*state.radius))

    def visual_action(tot_angle):
      return identity_visual_interaction(tot_angle[:, :, 0])

    dpos = ttc_visual_force_tot(state.position, state.velocity, state.radius, displacement, visual_action) + body_force(state.position) + goal_velocity_force_tot(state.velocity, state.goal_speed)
    return PedestrianState(dpos, None, None, None, None, None, None)

init, step = pedestrian(shift, force_fn, dt, N, stochastic=False)

# initialize
# theta = np.asin(2 * 0.1 / DIST) - 0.00000015
theta = 0
pos = np.array([np.array([1.5, frame_size/2]), np.array([1.5 + DIST, frame_size/2])])

velocity = np.array([2 * np.array([np.cos(theta), np.sin(theta)]), np.array([0.3, 0.])])

# print(pos.shape)
state = init(pos, 0.1, key=V_key, velocity=velocity, goal_speed=np.array([1.2, 0.3]))

print(state.velocity)

positions = []
velocities = []
thetas = []

for i in range(200):
  print(f"Current loop: {i}")
  state = lax.fori_loop(0, delta, step, state)

  positions += [state.position]
  velocities += [state.velocity]
  thetas += [state.orientation()]

print(state)

render(frame_size, positions, time_step, 'pedestrian_scattering_unseen', extra=thetas, limits=(0, 2 * np.pi), size=0.1)

frames = len(velocities)

velocities = np.array(velocities)
speeds = np.linalg.norm(velocities[:, 1], axis=1)

times = np.arange(0., time_step * (frames - 0.1), time_step)

plt.grid(True)
plt.xlim(0, dt * delta * frames)
plt.xlabel("Time (s)")
plt.ylim(0, 0.5)
plt.ylabel("Speed of second particle (m/s)")
plt.title("Speed over time")
plt.plot(times, speeds)

plt.savefig("unseen_spd_plots.png")
