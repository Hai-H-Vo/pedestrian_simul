# IMPORTS

import numpy as onp
import jax.numpy as np
import optax
from jax import random
from jax import jit
from jax import vmap
from jax import grad
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
from simulator.utils import normal, goal_velocity_force
from simulator.force import ttc_force_tot, wall_energy_tot, general_force_generator
from simulator.render import render
from simulator.dynamics import pedestrian, PedestrianState, StraightWall

# BASIS REPR

# POSITION
# TEST: ASSUME WE HAVE ARRAY OF POSITIONS, OF SHAPE (ITERATION_NUM, PARTICLE_NUM, DIM)
TIMESTEP = 1
ITER_NUM = 1000
PART_NUM = 100
DIM = 2
POSITIONS = np.zeros([ITER_NUM, PART_NUM, DIM])

# VELOCITY + ACCEL RETRIEVAL
VELOCITY = (POSITIONS[1:] - POSITIONS[:-1]) / TIMESTEP
ACCEL = (VELOCITY[1:] - VELOCITY[:-1]) / TIMESTEP

# PARAMS INIT
paral_weights = np.zeros([10, 10, 10])
perpen_weights = np.zeros([10, 10, 10])
d_0 = 10
v_0 = 10

def loss_fn(paral_weights, perpen_weights, d_0, v_0, pos, vel, accel):
    # GENERATED FORCE NEEDS TO BE VMAPPED
    # loss_fn = ||F_pred - F||^2
    return np.linalg.norm(general_force_generator(paral_weights, perpen_weights, v_0, d_0)(pos, vel, vel) - accel) ** 2

# OPTIMIZATION
start_learning_rate = 0.1
optimizer = optax.adam(start_learning_rate)

# NEEDS FIX - params structure
params = paral_weights
opt_state = optimizer.init(params)

# A simple update loop.
for _ in range(1000):
  grads = grad(loss_fn)(params, POSITIONS, VELOCITY, ACCEL)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
