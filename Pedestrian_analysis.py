from simulator.utils import ttc_tot, hmean, counter
from simulator.render import traj_render
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import vmap
from jax_md import space
from functools import partial


NAME = "SCENARIO_6_SESSION_1_TRIAL_2"

data = np.load(f"Pedestrians_Data/{NAME}.npz")

positions = data['positions']
velocities = data['velocities']
orientations = data['orientations']
speeds = data['speeds']
time_step = data['time_step']

traj_render(5.9, positions, NAME, origin=(-2.95, -2.95), null=np.array([999, 999]), focus=13)

# # plotting
# frames = np.shape(speeds)[0]
# total_num = np.shape(speeds)[1]
# times = np.arange(0., time_step * (frames - 0.1), time_step)
# print(np.shape(times))

# def counter(positions, null):
#     def _counter(position):
#         return np.sum(vmap(lambda pos: np.where(np.array_equal(pos, null), np.array(0), np.array(1)))(position))
#     return vmap(_counter)(positions)

# num_pedestrians = counter(positions, np.array([999, 999]))

# avg_speeds = np.sum(speeds, axis=1) / num_pedestrians

# avg_sq_speeds = np.sum(speeds * speeds, axis=1) / num_pedestrians

# displacement, shift = space.free()

# # print(ttc_tot(positions[0], velocities[0], 0.1, displacement)[0])

# def average_ttc_tot(position, velocity, radius, displacement):
#     ttc = ttc_tot(position, velocity, radius, displacement)
#     collision_num = np.sum(counter(ttc, np.array(999)))
#     return np.where(collision_num, hmean(ttc, where=(ttc!=np.array(999))), -1.)

# def average_ttc_time_tot(positions, velocities, radius, displacement):
#     return vmap(partial(average_ttc_tot, radius=radius, displacement=displacement))(positions, velocities)

# average_ttc = average_ttc_time_tot(positions, velocities, 0.1, displacement)
# print(average_ttc)

# plt.clf()

# plt.grid(True)
# plt.xlim(0, time_step * frames)
# plt.xlabel("Time (s)")
# plt.ylim(0, 12)
# plt.ylabel("Average collision time (s)")
# plt.title("Average collision time (harmonic mean) over time")
# plt.plot(times, average_ttc)

# plt.savefig(f"Pedestrians_Data/{NAME}_harmonic_ttc_plots.png")

# def average_ttc_tot(position, velocity, radius, displacement):
#     pedestrian_num = np.shape(position)[0]
#     ttc = ttc_tot(position, velocity, radius, displacement)
#     collision_num = np.sum(counter(ttc, np.array(999)))
#     return np.where(collision_num, (np.sum(ttc) - (pedestrian_num ** 2 - collision_num) * 9.) / collision_num, -1.)

# def average_ttc_time_tot(positions, velocities, radius, displacement):
#     return vmap(partial(average_ttc_tot, radius=radius, displacement=displacement))(positions, velocities)

# average_ttc = average_ttc_time_tot(positions, velocities, 0.1, displacement)
# print(average_ttc)

# plt.grid(True)
# plt.xlim(0, time_step * frames)
# plt.xlabel("Time (s)")
# plt.ylim(0, np.max(average_ttc) * 1.1)
# plt.ylabel("Average collision time (s)")
# plt.title("Average collision time over time")
# plt.plot(times, average_ttc)

# plt.savefig(f"Pedestrians_Data/{NAME}_ttc_amean_plots.png")
