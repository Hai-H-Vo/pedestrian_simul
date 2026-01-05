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
from simulator.force import ttc_visual_force_tot, ttc_visual_force_unsummed_tot, wall_energy_tot, goal_velocity_force_tot
from simulator.vision import identity_visual_interaction_tot
from simulator.render import render
from simulator.dynamics import pedestrian, PedestrianState, StraightWall


N = 300
dt = 0.005
delta = 50
box_size = 40
hall_width = 20
key = random.PRNGKey(0)
displacement, shift = space.periodic(box_size)
# displacement, shift = space.free()

V_key, pos_key = random.split(key)

ll = np.array([0., (box_size - hall_width) / 2])
ul = np.array([0., (box_size + hall_width) / 2])
lr = np.array([box_size, (box_size - hall_width) / 2])
ur = np.array([box_size, (box_size + hall_width) / 2])

wall_low = StraightWall(ll, lr)
wall_up = StraightWall(ul, ur)

def energy_fn(pos, radius):
    return wall_energy_tot(pos, wall_low, radius, displacement) + wall_energy_tot(pos, wall_up, radius, displacement)

def force_fn(state):
    wall_force = quantity.force(partial(energy_fn, radius=state.radius))
    body_force = quantity.force(energy.soft_sphere_pair(displacement, sigma=2.*state.radius))
    return PedestrianState(ttc_visual_force_tot(state.position, state.velocity, state.radius, displacement, identity_visual_interaction_tot, 1.5, 3)
                          + body_force(state.position) + wall_force(state.position) + goal_velocity_force_tot(state.velocity, state.goal_speed, state.goal_orientation), None, None, None, None, None, None)

init, step = pedestrian(shift, force_fn, dt, N)

# position initialize
# 15 * 10 * 2 = N = 300
y_start = np.linspace((box_size - hall_width) / 2 + 0.5, (box_size + hall_width) / 2 - 0.5, 15)
x_start_1 = np.linspace(0.5, box_size / 4, 10)
x_start_2 = np.linspace(3 * box_size / 4, box_size - 0.5, 10)

# y_start = np.linspace((box_size - hall_width) / 2 + 0.5, (box_size + hall_width) / 2 - 0.5, 4)
# x_start_1 = np.linspace(0.5, box_size / 4, 3)
# x_start_2 = np.linspace(3 * box_size / 4, box_size - 0.5, 3)

pos_1_x, pos_1_y = np.meshgrid(x_start_1, y_start, copy=True)
pos_1 = np.stack((pos_1_x.reshape(-1), pos_1_y.reshape(-1)), axis=1)

pos_2_x, pos_2_y = np.meshgrid(x_start_2, y_start, copy=True)
pos_2 = np.stack((pos_2_x.reshape(-1), pos_2_y.reshape(-1)), axis=1)

pos = np.concatenate((pos_1, pos_2))

# print(pos.shape)

# orientation initialize
goal_orientation = np.concatenate((np.zeros((150,)), onp.pi * np.ones((150,))))

state = init(pos, 0.1, key=V_key, goal_orientation=goal_orientation)

positions = []
velocities = []
thetas = []
seen = []

for i in range(1250):
  print(f"Current loop: {i}")
  state = lax.fori_loop(0, delta, step, state)

  positions += [state.position]
  velocities += [state.velocity]
  thetas += [state.goal_orientation]
  seen += [np.linalg.norm(ttc_visual_force_unsummed_tot(state.position, state.velocity, state.radius, displacement, identity_visual_interaction_tot, 1.5, 3), axis=2)]

# MP4 PRODUCTION
render(box_size, positions, dt * delta, 'pedestrian_hallway', extra=thetas, limits=(0, 2 * onp.pi), walls=[wall_low, wall_up], vision_target=150, detections=np.array(seen), threshold=0.001)

# # NPZ PRODUCTION
np_positions = np.array(positions)
np_velocities = np.array(velocities)
np_orientations = np.array(thetas)
time_step = np.array(delta * dt)

np_wall_low = np.array([ll, lr])
np_wall_up = np.array([ul, ur])
np_walls = np.array([np_wall_low, np_wall_up])

np.savez("pedestrian_hallway", positions = np_positions, velocities = np_velocities, goal_orientations = np_orientations, walls = np_walls, time_step=time_step)
