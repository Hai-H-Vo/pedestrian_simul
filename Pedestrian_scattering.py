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
# config.update('jax_traceback_filtering', 'off')

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

from collections import namedtuple

vectorize = np.vectorize

from functools import partial
from simulator.utils import normal, goal_velocity_force, _ttc_force_tot
from simulator.force import ttc_force_tot
from simulator.render import render
from simulator.dynamics import pedestrian, PedestrianState, StraightWall

# YOUR CHOICE!
DIST = 3.

N = 2
dt = 0.001
delta = 20
frame_size = 5
key = random.PRNGKey(0)
# displacement, shift = space.periodic(box_size)
displacement, shift = space.free()

V_key, pos_key = random.split(key)

def force_fn(state):
    body_force = quantity.force(energy.soft_sphere_pair(displacement, sigma=2.*state.radius))
    dpos = _ttc_force_tot(state.position, state.velocity, state.radius, displacement, 1.5, 3) + body_force(state.position) + goal_velocity_force(state)
    dpos = np.array([dpos[0], np.array([0., 0.])])
    return PedestrianState(dpos, None, None, None, None, None)

init, step = pedestrian(shift, force_fn, dt, N, stochastic=False)

# initialize
theta = np.asin(2 * 0.1 / DIST) - 0.00000015
pos = np.array([np.array([1., frame_size/2]), np.array([1. + DIST, frame_size/2])])
velocity = np.array([np.array([np.cos(theta), np.sin(theta)]), np.array([0., 0.])])

# print(pos.shape)
state = init(pos, 0.1, key=V_key, velocity=velocity)

positions = []
thetas = []

for i in range(200):
  print(f"Current loop: {i}")
  state = lax.fori_loop(0, delta, step, state)

  positions += [state.position]
  thetas += [state.orientation()]

print(state)

render(frame_size, positions, dt * delta, 'pedestrian_scattering', extra=thetas, limits=(0, 2 * onp.pi), size=0.1)
