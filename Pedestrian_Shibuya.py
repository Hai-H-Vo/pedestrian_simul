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

# N can be 3000
N = 600
dt = 0.002
delta = 50

unit = 7
rel_frame_size = (27 + 2 * np.sqrt(2)) / 4
frame_size = rel_frame_size * unit

key = random.PRNGKey(0)
displacement, shift = space.periodic(frame_size)
# displacement, shift = space.free()

V_key, pos_key = random.split(key)

# goals

# 4 corners
goal_1 = unit * np.array([0.75, rel_frame_size - 0.75])
goal_2 = unit * np.array([rel_frame_size - 0.75, rel_frame_size - 0.75])
goal_3 = unit * np.array([rel_frame_size - 0.75, 0.75])
goal_4 = unit * np.array([0.75, 0.75])

# goal directions
goal_5 = unit * np.array([-690., rel_frame_size - 0.75])
goal_6 = unit * np.array([0.75, 690.])
goal_7 = unit * np.array([rel_frame_size - 0.75, 690.])
goal_8 = unit * np.array([690., rel_frame_size - 0.75])
goal_9 = unit * np.array([690., 0.75])
goal_10 = unit * np.array([rel_frame_size - 0.75, -690.])
goal_11 = unit * np.array([0.75, -690.])
goal_12 = unit * np.array([-690., 0.75])

goals = np.array([goal_1, goal_2, goal_3, goal_4, goal_5, goal_6, goal_7, goal_8, goal_9, goal_10, goal_11, goal_12])
goal_rad = 0.9 * unit * 0.5 * np.tan(3 * np.pi / 8)
far_goal_rad = unit * rel_frame_size / 3

# transition rules:
# can be improved!
def _cond_fn_1(pos, goal, key):
    disp_to_goal = goal - pos
    dist_to_goal = np.linalg.norm(disp_to_goal)
    return np.where(dist_to_goal > goal_rad, goal_1,
                    random.choice(key, np.array([goal_5, goal_6, goal_2, goal_4])))

def _cond_fn_2(pos, goal, key):
    disp_to_goal = goal - pos
    dist_to_goal = np.linalg.norm(disp_to_goal)
    return np.where(dist_to_goal > 0.75, goal_2,
                    random.choice(key, np.array([goal_7, goal_8])))

def _cond_fn_3(pos, goal, key):
    disp_to_goal = goal - pos
    dist_to_goal = np.linalg.norm(disp_to_goal)
    return np.where(dist_to_goal > goal_rad, goal_3,
                    random.choice(key, np.array([goal_9, goal_10, goal_2, goal_4])))

def _cond_fn_4(pos, goal, key):
    disp_to_goal = goal - pos
    dist_to_goal = np.linalg.norm(disp_to_goal)
    return np.where(dist_to_goal > 0.75, goal_4,
                    random.choice(key, np.array([goal_11, goal_12])))

def _cond_fn_5_6(pos, goal, key):
    disp_to_goal = goal_1 - pos
    dist_to_goal = np.linalg.norm(disp_to_goal)
    return np.where(dist_to_goal < far_goal_rad, goal,
                    random.choice(key, np.array([goal_1, goal_3])))

def _cond_fn_7(pos, goal, key):
    disp_to_goal = goal_2 - pos
    dist_to_goal = np.linalg.norm(disp_to_goal)
    return np.where(dist_to_goal < far_goal_rad, goal,
                    random.choice(key, np.array([goal_1, goal_2, goal_4])))

def _cond_fn_8(pos, goal, key):
    disp_to_goal = goal_2 - pos
    dist_to_goal = np.linalg.norm(disp_to_goal)
    return np.where(dist_to_goal < far_goal_rad, goal,
                    random.choice(key, np.array([goal_2, goal_3, goal_4])))

def _cond_fn_9_10(pos, goal, key):
    disp_to_goal = goal_3 - pos
    dist_to_goal = np.linalg.norm(disp_to_goal)
    return np.where(dist_to_goal < far_goal_rad, goal,
                    random.choice(key, np.array([goal_1, goal_3])))

def _cond_fn_11(pos, goal, key):
    disp_to_goal = goal_4 - pos
    dist_to_goal = np.linalg.norm(disp_to_goal)
    return np.where(dist_to_goal < far_goal_rad, goal,
                    random.choice(key, np.array([goal_2, goal_3, goal_4])))

def _cond_fn_12(pos, goal, key):
    disp_to_goal = goal_4 - pos
    dist_to_goal = np.linalg.norm(disp_to_goal)
    return np.where(dist_to_goal < far_goal_rad, goal,
                    random.choice(key, np.array([goal_1, goal_2, goal_4])))


# walls
rel_sidewalk_width = 1.25 + 1 / np.sqrt(2)

hori1 = unit * np.array([1.1 * rel_sidewalk_width, 0])
hori2 = unit * np.array([rel_frame_size - 1.1 * rel_sidewalk_width, 0])

hori3 = unit * np.array([1.1 * rel_sidewalk_width, rel_frame_size])
hori4 = unit * np.array([rel_frame_size - 1.1 * rel_sidewalk_width, rel_frame_size])

vert1 = unit * np.array([0, 1.1 * rel_sidewalk_width])
vert2 = unit * np.array([0, rel_frame_size - 1.1 * rel_sidewalk_width])

vert3 = unit * np.array([rel_frame_size, 1.1 * rel_sidewalk_width])
vert4 = unit * np.array([rel_frame_size, rel_frame_size - 1.1 * rel_sidewalk_width])

wall_vert_l = StraightWall(vert1, vert2)
wall_vert_r = StraightWall(vert3, vert4)
wall_hori_l = StraightWall(hori1, hori2)
wall_hori_u = StraightWall(hori3, hori4)

def energy_fn(pos, radius):
    return (wall_energy_tot(pos, wall_vert_l, radius, displacement) +
            wall_energy_tot(pos, wall_vert_r, radius, displacement) +
            wall_energy_tot(pos, wall_hori_l, radius, displacement) +
            wall_energy_tot(pos, wall_hori_u, radius, displacement))

def force_fn(state):
    wall_force = quantity.force(partial(energy_fn, radius=state.radius))
    body_force = quantity.force(energy.soft_sphere_pair(displacement, sigma=2.*state.radius))

    # goals = np.array([goal_point, np.array([369., frame_size/2])])

    # # @vmap
    # def cond_fn_0(pos, goal, goals):
    #     disp_to_goal = goal - pos
    #     dist_to_goal = np.linalg.norm(disp_to_goal)
    #     return np.where(dist_to_goal > 0.1, goals[0], goals[1])

    # # @vmap
    # def cond_fn_1(_, __, goals):
    #     return goals[1]

    # new_goal_fn = dgoal_generator([cond_fn_0, cond_fn_1], np.array([goal_point, np.array([369., frame_size/2])]))

    # new_goal = new_goal_fn(state.position, state.goal)
    key = state.key
    conditions = [_cond_fn_1, _cond_fn_2, _cond_fn_3, _cond_fn_4, _cond_fn_5_6, _cond_fn_5_6, _cond_fn_7, _cond_fn_8, _cond_fn_9_10, _cond_fn_9_10, _cond_fn_11, _cond_fn_12]
    transitions = []
    for cond in conditions:
        key, split = random.split(key)
        transitions.append(partial(cond, key=split))
    new_goal_fn = dgoal_generator(transitions, goals)

    new_goal = new_goal_fn(state.position, state.goal)

    return ExperimentalPedestrianState(ttc_force_tot(state.position, state.velocity, state.radius, displacement, 1.5, 3) +
                           body_force(state.position) + wall_force(state.position) +
                           goal_velocity_force_tot(state.velocity, state.goal_speed, state.goal_orientation),
                           None, None, None, None, new_goal, None)


init, step = pedestrian(shift, force_fn, dt, N, experimental=True)

# position initialize
pos_key_1, pos_key_2, pos_key_3, pos_key_4, goal_key = random.split(pos_key, 5)

quadrant_num = int(N/4)
corner_shift = unit * np.array([-0.5, -0.5])
pos_1 = (goal_1 + corner_shift) + (unit * random.uniform(pos_key_1, (quadrant_num, 2)))
pos_2 = (goal_2 + corner_shift) + (unit * random.uniform(pos_key_2, (quadrant_num, 2)))
pos_3 = (goal_3 + corner_shift) + (unit * random.uniform(pos_key_3, (quadrant_num, 2)))
pos_4 = (goal_4 + corner_shift) + (unit * random.uniform(pos_key_4, (quadrant_num, 2)))
pos = np.concatenate((pos_1, pos_2, pos_3, pos_4), axis=0)

# goal initialize:
goal_key_1, goal_key_2, goal_key_3, goal_key_4 = random.split(goal_key, 4)
goal_list_1 = random.choice(goal_key_1, np.array([goal_2, goal_3, goal_4]), (quadrant_num,))
goal_list_2 = random.choice(goal_key_2, np.array([goal_1, goal_3]), (quadrant_num,))
goal_list_3 = random.choice(goal_key_3, np.array([goal_1, goal_2, goal_4]), (quadrant_num,))
goal_list_4 = random.choice(goal_key_4, np.array([goal_1, goal_3]), (quadrant_num,))

goal = np.concatenate((goal_list_1, goal_list_2, goal_list_3, goal_list_4), axis=0)

# initialize
state = init(pos, 0.1, key=V_key, goal=goal)

positions = []
thetas = []

# 1250
for i in range(1500):
    print(f"Current loop: {i}")
    state = lax.fori_loop(0, delta, step, state)

    positions += [state.position]
    thetas += [state.orientation()]

# print(state)

# DRAWING - OPTIONAL
lines = []

# sidewalks
point_1 = unit * np.array([rel_sidewalk_width, 0])
point_2 = unit * np.array([rel_sidewalk_width, rel_sidewalk_width])
point_3 = unit * np.array([0, rel_sidewalk_width])

point_4 = unit * np.array([rel_frame_size - rel_sidewalk_width, 0])
point_5 = unit * np.array([rel_frame_size - rel_sidewalk_width, 1.25])
point_6 = unit * np.array([rel_frame_size - 1.25, rel_sidewalk_width])
point_7 = unit * np.array([rel_frame_size, rel_sidewalk_width])

point_8 = unit * np.array([rel_frame_size - rel_sidewalk_width, rel_frame_size])
point_9 = unit * np.array([rel_frame_size - rel_sidewalk_width, rel_frame_size - rel_sidewalk_width])
point_10 = unit * np.array([rel_frame_size, rel_frame_size - rel_sidewalk_width])

point_11 = unit * np.array([rel_sidewalk_width, rel_frame_size])
point_12 = unit * np.array([rel_sidewalk_width, rel_frame_size - 1.25])
point_13 = unit * np.array([1.25, rel_frame_size - rel_sidewalk_width])
point_14 = unit * np.array([0, rel_frame_size - rel_sidewalk_width])

lines.extend([np.array([point_1, point_2]), np.array([point_2, point_3])])
lines.extend([np.array([point_4, point_5]), np.array([point_5, point_6]), np.array([point_6, point_7])])
lines.extend([np.array([point_8, point_9]), np.array([point_9, point_10])])
lines.extend([np.array([point_11, point_12]), np.array([point_12, point_13]), np.array([point_13, point_14])])

# crossings
point_15 = unit * np.array([rel_sidewalk_width, 0.25])
point_16 = unit * np.array([rel_sidewalk_width, 1.25])
point_17 = unit * np.array([1.25, rel_sidewalk_width])
point_18 = unit * np.array([0.25, rel_sidewalk_width])

point_19 = unit * np.array([rel_frame_size - rel_sidewalk_width, 0.25])
point_20 = unit * np.array([rel_frame_size - 0.25, rel_sidewalk_width])

point_21 = unit * np.array([rel_frame_size - 0.25, rel_frame_size - rel_sidewalk_width])
point_22 = unit * np.array([rel_frame_size - 1.25, rel_frame_size - rel_sidewalk_width])
point_23 = unit * np.array([rel_frame_size - rel_sidewalk_width, rel_frame_size - 1.25])
point_24 = unit * np.array([rel_frame_size - rel_sidewalk_width, rel_frame_size - 0.25])

point_25 = unit * np.array([rel_sidewalk_width, rel_frame_size - 0.25])
point_26 = unit * np.array([0.25, rel_frame_size - rel_sidewalk_width])

lines.extend([np.array([point_26, point_18]), np.array([point_13, point_17])])
lines.extend([np.array([point_12, point_6]), np.array([point_13, point_5])])
lines.extend([np.array([point_12, point_23]), np.array([point_25, point_24])])

lines.extend([np.array([point_16, point_5]), np.array([point_15, point_19])])
lines.extend([np.array([point_6, point_22]), np.array([point_20, point_21])])

# MP4 PRODUCTION
render(frame_size, positions, dt * delta, 'pedestrian_shibuya_experimental', extra=thetas, limits=(0, 2 * onp.pi), walls=[wall_hori_l, wall_hori_u, wall_vert_l, wall_vert_r], lines=lines, size=0.1)

# NPZ PRODUCTION
np_positions = np.array(positions)
np_orientations = np.array(thetas)
time_step = np.array(delta * dt)

walls = [numpify_wall(wall) for wall in (wall_hori_l, wall_hori_u, wall_vert_l, wall_vert_r)]
np_walls = np.array(walls)

np.savez("pedestrian_shibuya_experimental", positions = np_positions, orientations = np_orientations, walls = np_walls, time_step=time_step)
