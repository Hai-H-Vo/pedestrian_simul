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
from simulator.utils import normal, numpify_wall, dgoal_generator
from simulator.force import ttc_force_tot, wall_energy_tot, goal_velocity_force_tot
from simulator.render import render
from simulator.dynamics import pedestrian, ExperimentalPedestrianState, StraightWall


N = 150
dt = 0.001
delta = 20
frame_size = 36

room_length = 24
room_width = 10
doorway_width = 0.4
key = random.PRNGKey(0)
displacement, shift = space.free()
# displacement, shift = space.free()

V_key, pos_key = random.split(key)

# room corners
ll = np.array([0., (frame_size - room_width) / 2])
ul = np.array([0., (frame_size + room_width) / 2])
lr = np.array([room_length, (frame_size - room_width) / 2])
ur = np.array([room_length, (frame_size + room_width) / 2])
# door corners
ld = np.array([room_length, (frame_size - doorway_width) / 2])
ud = np.array([room_length, (frame_size + doorway_width) / 2])

wall_left = StraightWall(ll, lr)
wall_upper = StraightWall(ul, ur)
wall_lower = StraightWall(ll, lr)
wall_right_up = StraightWall(ur, ud)
wall_right_down = StraightWall(lr, ld)

# goal point
goal_point = np.array([room_length, frame_size / 2])

def energy_fn(pos, radius):
    return (wall_energy_tot(pos, wall_left, radius, displacement) +
            wall_energy_tot(pos, wall_lower, radius, displacement) +
            wall_energy_tot(pos, wall_upper, radius, displacement) +
            wall_energy_tot(pos, wall_right_down, radius, displacement) +
            wall_energy_tot(pos, wall_right_up, radius, displacement) )


def force_fn(state):
    wall_force = quantity.force(partial(energy_fn, radius=state.radius))
    body_force = quantity.force(energy.soft_sphere_pair(displacement, sigma=2.*state.radius))

    # goals = np.array([goal_point, np.array([369., frame_size/2])])

    # @vmap
    def cond_fn_0(pos, goal):
        disp_to_goal = goal - pos
        dist_to_goal = np.linalg.norm(disp_to_goal)
        return np.where(dist_to_goal > 0.1, goal_point, np.array([369., frame_size/2]))

    # @vmap
    def cond_fn_1(_, __):
        return np.array([369., frame_size/2])

    new_goal_fn = dgoal_generator([cond_fn_0, cond_fn_1], np.array([goal_point, np.array([369., frame_size/2])]))

    new_goal = new_goal_fn(state.position, state.goal)

    return ExperimentalPedestrianState(ttc_force_tot(state.position, state.velocity, state.radius, displacement, 1.5, 3) +
                           body_force(state.position) + wall_force(state.position) +
                           goal_velocity_force_tot(state.velocity, state.goal_speed, state.goal_orientation),
                           None, None, None, None, new_goal, None)


init, step = pedestrian(shift, force_fn, dt, N, experimental=True)

# position initialize
pos_key_1, pos_key_2 = random.split(pos_key)

pos_0 = np.array([ll] * N)
pos = pos_0 + (0.5 + np.array([3 * room_length / 4, room_width - 1]) * random.uniform(pos_key, (N, 2)))

# print(pos.shape)

# goal initialize:
goal = np.array([goal_point] * N)

# initialize
state = init(pos, 0.1, key=V_key, goal=goal)

# state = init(pos, 0.1, key=V_key, goal_orientation=goal_orientation)

positions = []
thetas = []

for i in range(1250):
    print(f"Current loop: {i}")
    # for _ in range(0, delta):
    #     state = step(_, state)
    state = lax.fori_loop(0, delta, step, state)

    positions += [state.position]
    thetas += [state.orientation()]

print(state)

# MP4 PRODUCTION
render(frame_size, positions, dt * delta, 'pedestrian_evacuation_experimental', extra=thetas, limits=(0, 2 * onp.pi), walls=[wall_upper, wall_lower, wall_left, wall_right_down, wall_right_up], size=0.1)

# NPZ PRODUCTION
np_positions = np.array(positions)
np_orientations = np.array(thetas)
time_step = np.array(delta * dt)

walls = [numpify_wall(wall) for wall in (wall_upper, wall_lower, wall_left, wall_right_down, wall_right_up)]
np_walls = np.array(walls)

np.savez("pedestrian_evacuation_experimental", positions = np_positions, orientations = np_orientations, walls = np_walls, time_step=time_step)
