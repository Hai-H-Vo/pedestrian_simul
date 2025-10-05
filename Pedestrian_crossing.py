# IMPORTS

import numpy as onp
import jax.numpy as np
from jax import random
from jax import jit
from jax import vmap
from jax import lax
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_debug_nans', True)
config.update('jax_traceback_filtering', 'off')

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

from collections import namedtuple

vectorize = np.vectorize

from functools import partial
from simulator.utils import normal
from simulator.force import ttc_force_tot, goal_velocity_force_tot
from simulator.render import render
from simulator.dynamics import pedestrian, PedestrianState, StraightWall


N = 80
dt = 0.001
delta = 20
frame_size = 40
lane_width = 5
key = random.PRNGKey(0)
displacement, shift = space.free()

V_key, pos_key = random.split(key)

def force_fn(state):
    body_force = quantity.force(energy.soft_sphere_pair(displacement, sigma=2.*state.radius))
    return PedestrianState(ttc_force_tot(state.position, state.velocity, state.radius, displacement, 1.5, 3) + body_force(state.position) + goal_velocity_force_tot(state.velocity, state.goal_speed, state.goal_orientation), None, None, None, None, None)

init, step = pedestrian(shift, force_fn, dt, N)

# position initialize
width_start = np.linspace((frame_size - lane_width) / 2, (frame_size + lane_width) / 2, 5)
length_start = np.linspace(0, frame_size / 4, 8)

pos_1_x, pos_1_y = np.meshgrid(width_start, length_start, copy=True)
pos_1 = np.stack((pos_1_x.reshape(-1), pos_1_y.reshape(-1)), axis=1)

pos_2_x, pos_2_y = np.meshgrid(length_start, width_start, copy=True)
pos_2 = np.stack((pos_2_x.reshape(-1), pos_2_y.reshape(-1)), axis=1)

pos = np.concatenate((pos_1, pos_2))

# print(pos.shape)

# orientation initialize
goal_orientation = np.concatenate(((onp.pi / 2) * np.ones((40,)), np.zeros((40,))))

state = init(pos, 0.1, key=V_key, goal_orientation=goal_orientation)

positions = []
thetas = []

for i in range(1500):
  print(f"Current loop: {i}")
  state = lax.fori_loop(0, delta, step, state)

  positions += [state.position]
  thetas += [state.goal_orientation]

print(state)

# MP4 PRODUCTION
render(frame_size, positions, dt * delta, 'pedestrian_crossing', extra=thetas, limits=(0, 2 * onp.pi))

# # NPZ PRODUCTION
# np_positions = np.array(positions)
# np_orientations = np.array(thetas)
# time_step = np.array(delta * dt)
# np.savez("pedestrian_crossing", positions = np_positions, goal_orientations = np_orientations, time_step=time_step)
