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
from simulator.force import ttc_force_tot, wall_energy_tot, goal_velocity_force_tot
from simulator.render import render
from simulator.dynamics import pedestrian, PedestrianState, StraightWall


N = 750
dt = 0.005
delta = 50
box_size = 40
key = random.PRNGKey(0)
displacement, shift = space.periodic(box_size)
# displacement, shift = space.free()

V_key, pos_key = random.split(key)

ll = np.array([0., 0.])
ul = np.array([0., box_size])
lr = np.array([box_size, 0.])
ur = np.array([box_size, box_size])
wall1 = StraightWall(ll, ul)
wall2 = StraightWall(ll, lr)
wall3 = StraightWall(ul, ur)
wall4 = StraightWall(lr, ur)

def energy_fn(pos, radius):
    return wall_energy_tot(pos, wall1, radius, displacement) + wall_energy_tot(pos, wall2, radius, displacement) + wall_energy_tot(pos, wall3, radius, displacement) + wall_energy_tot(pos, wall4, radius, displacement)

def force_fn(state):
    wall_force = quantity.force(partial(energy_fn, radius=state.radius))
    body_force = quantity.force(energy.soft_sphere_pair(displacement, sigma=2.*state.radius))
    return PedestrianState(ttc_force_tot(state.position, state.velocity, state.radius, displacement, 1.5, 3) + body_force(state.position) + wall_force(state.position) + goal_velocity_force_tot(state.velocity, state.goal_speed, state.goal_orientation), None, None, None, None, None)

init, step = pedestrian(shift, force_fn, dt, N)

pos = 0.5 + (box_size - 1) * random.uniform(pos_key, (N, 2))

state = init(pos, 0.1, key=V_key)

positions = []
thetas = []

for i in range(2000):
  print(f"Current loop: {i}")
  state = lax.fori_loop(0, delta, step, state)

  positions += [state.position]
  thetas += [state.orientation()]

print(state)

# MP4 PRODUCTION
render(box_size, positions, dt * delta, 'pedestrian_test', extra=thetas, limits=(0, 2 * onp.pi))

# NPY PRODUCTION

np_positions = np.array(positions)
np_orientations = np.array(thetas)
time_step = np.array(delta * dt)

np_wall1 = np.array([ll, ul])
np_wall2 = np.array([ll, lr])
np_wall3 = np.array([ul, ur])
np_wall4 = np.array([lr, ur])
np_walls = np.array([np_wall1, np_wall2, np_wall3, np_wall4])

np.savez("pedestrian_collective", positions = np_positions, orientations = np_orientations, walls = np_walls, time_step = time_step)
