from simulator.extract import extract
from simulator.render import render
import matplotlib.pyplot as plt
import jax.numpy as np

NAME = "SCENARIO_1_SESSION_1_TRIAL_1"

data = extract(f"simulator/{NAME}")

orientations = data['orientations']
speeds = data['speeds']
time_step = data['time_step']
spd_max = np.max(speeds)

# spd min:
# zeros = int(np.unique(speeds, return_counts=True)[1][0])
# spd_min = np.partition(np.hstack(speeds), zeros + 1)[zeros]

# adjust for each simul
# render(5.9, data['positions'], data['time_step'], name=NAME, origin=(-2.9, -2.9), size=0.1, extra=speeds, limits=(0, spd_max), periodic=False)
render(5.9, data['positions'], data['time_step'], name=NAME, origin=(-2.9, -2.9), size=0.1, extra=orientations, limits=(-np.pi, np.pi), periodic=True)

# plotting
frames = np.shape(speeds)[0]
times = np.arange(0., time_step * frames, time_step)

avg_speeds = np.sum(speeds, axis=1) / np.where(np.count_nonzero(speeds, axis=1), np.count_nonzero(speeds, axis=1), 1)

plt.xlim(0, time_step * frames)
plt.xlabel("Time (s)")
plt.ylim(0, 1.05 * spd_max)
plt.ylabel("Average speed (m/s)")
plt.plot(times, avg_speeds)
plt.savefig(f"{NAME}_avg_spd_plot.png")
