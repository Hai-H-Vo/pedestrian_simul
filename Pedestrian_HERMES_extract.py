from simulator.extract import extract, load_hermes
from simulator.render import render
from simulator.force import ttc_visual_force_unsummed_tot
from simulator.vision import identity_visual_interaction_tot
from simulator.utils import ttc_tot, hmean
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import vmap
from jax_md import space
from functools import partial

# NAME = "bo-360-050-050"
NAME = "uo-050-180-180"

SMOOTH = False

SAVE_NAME = NAME if not SMOOTH else f"{NAME}_smooth"

load_hermes(f"HERMES/{NAME}.txt")

data = extract(f"HERMES/{NAME}", save=True, savefilename=f"Pedestrians_Data/{SAVE_NAME}", smooth=SMOOTH, timeframe=9)

print('hi')
positions = data['positions']
velocities = data['velocities']
orientations = data['orientations']
speeds = data['speeds']
time_step = data['time_step']

# spd min:
# zeros = int(np.unique(speeds, return_counts=True)[1][0])
# spd_min = np.partition(np.hstack(speeds), zeros + 1)[zeros]

# adjust for each simul
# render(5.9, data['positions'], data['time_step'], name=NAME, origin=(-2.9, -2.9), size=0.1, extra=speeds, limits=(0, spd_max), periodic=False)

# plotting
frames = np.shape(speeds)[0]
total_num = np.shape(speeds)[1]
times = np.arange(0., time_step * (frames - 0.1), time_step)
print(np.shape(times))

# interaction graphing:
displacement, shift = space.free()

detections = np.linalg.norm(vmap(partial(ttc_visual_force_unsummed_tot, visual_action=identity_visual_interaction_tot, k=1.5, t_0=3.0), (0, 0, None, None))(positions, velocities, 0.1, displacement), axis=3)

render(14, positions, time_step, name=f"Pedestrians_Data/{SAVE_NAME}", origin=(-5., -7.), size=0.1, extra=orientations, limits=(-np.pi, np.pi), periodic=True,
       vision_target=int(total_num / 4), detections=detections, threshold=0.001)

def counter(positions, null):
    def _counter(position):
        return np.sum(vmap(lambda pos: np.where(np.array_equal(pos, null), np.array(0), np.array(1)))(position))
    return vmap(_counter)(positions)

num_pedestrians = counter(positions, np.array([999, 999]))

avg_speeds = np.sum(speeds, axis=1) / num_pedestrians

avg_sq_speeds = np.sum(speeds * speeds, axis=1) / num_pedestrians



# print(ttc_tot(positions[0], velocities[0], 0.1, displacement)[0])

def average_ttc_tot(position, velocity, radius, displacement):
    ttc = ttc_tot(position, velocity, radius, displacement)
    collision_num = np.sum(counter(ttc, np.array(999)))
    return np.where(collision_num, hmean(ttc, where=(ttc!=np.array(999))), -1.)

def average_ttc_time_tot(positions, velocities, radius, displacement):
    return vmap(partial(average_ttc_tot, radius=radius, displacement=displacement))(positions, velocities)

average_ttc = average_ttc_time_tot(positions, velocities, 0.1, displacement)
print(average_ttc)

plt.grid(True)
plt.xlim(0, time_step * frames)
plt.xlabel("Time (s)")
plt.ylim(0, np.max(avg_speeds) * 1.1)
plt.ylabel("Average speed (m/s)")
plt.title("Average speed of pedestrians over time")
plt.plot(times, avg_speeds)

plt.savefig(f"Pedestrians_Data/{SAVE_NAME}_spd_plots.png")

plt.clf()

plt.grid(True)
plt.xlim(0, time_step * frames)
plt.xlabel("Time (s)")
plt.ylim(0, np.max(avg_sq_speeds) * 1.1)
plt.ylabel("Average squared speed (m^2/s^2)")
plt.title("Average squared speed of pedestrians over time")
plt.plot(times, avg_sq_speeds)

plt.savefig(f"Pedestrians_Data/{SAVE_NAME}_energy_plots.png")

plt.clf()

plt.grid(True)
plt.xlim(0, time_step * frames)
plt.xlabel("Time (s)")
plt.ylim(0, 12)
plt.ylabel("Average collision time (s)")
plt.title("Average collision time (harmonic mean) over time")
plt.plot(times, average_ttc)

plt.savefig(f"Pedestrians_Data/{SAVE_NAME}_harmonic_ttc_plots.png")
