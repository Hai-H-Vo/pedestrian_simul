# synth_ped_dataset_with_export.py
# Generate a two-group pedestrian dataset (JAX), save to NPZ, and export an MP4 (or GIF) animation.

from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# ----------------------------- Simulation core -----------------------------

@dataclass
class SimParams:
    # Interaction (social force) parameters
    A: float = 5.0
    B: float = 1.0
    d0: float = 0.5
    r_cut: float = 5.0

    # Driving force toward target velocity
    tau: float = 0.6

    # Noise
    sigma: float = 0.1

    # Integration
    dt: float = 0.05
    mass: float = 1.0

    # Box / initialization
    x_left: float = -8.0
    x_right: float = 8.0
    y_min: float = -4.0
    y_max: float = 4.0
    jitter_xy: float = 0.5

    # Target speeds
    v_target_mag: float = 1.5
    v_target_rel_jitter: float = 0.1


def _pairwise_repulsion(x: jnp.ndarray, p: SimParams) -> jnp.ndarray:
    N = x.shape[0]
    diff = x[:, None, :] - x[None, :, :]      # [N, N, 2]
    d2 = jnp.sum(diff * diff, axis=-1)        # [N, N]
    eps = 1e-6
    d = jnp.sqrt(d2 + eps)                    # [N, N]
    n_ij = diff / d[..., None]                # [N, N, 2]
    eye = jnp.eye(N, dtype=jnp.float32)
    mag = p.A * jnp.exp((p.d0 - d) / p.B) * (1.0 - eye)
    if p.r_cut > 0.0:
        mag = mag * (d <= p.r_cut)
    F = jnp.sum(mag[..., None] * n_ij, axis=1)  # [N, 2]
    return F


def _driving_force(v: jnp.ndarray, v_target: jnp.ndarray, p: SimParams) -> jnp.ndarray:
    return (v_target - v) / p.tau


def _semi_implicit_euler(x: jnp.ndarray, v: jnp.ndarray, a: jnp.ndarray, p: SimParams):
    v_next = v + p.dt * a
    x_next = x + p.dt * v_next
    return x_next, v_next


def _step(carry, rng, v_target, p: SimParams):
    x, v = carry
    F_rep = _pairwise_repulsion(x, p)
    F_drv = _driving_force(v, v_target, p)
    F_tot = F_rep + F_drv

    if p.sigma > 0.0:
        rng, k_noise = jax.random.split(rng)
        noise = jax.random.normal(k_noise, v.shape) * (p.sigma / jnp.sqrt(p.mass)) * jnp.sqrt(p.dt)
    else:
        noise = jnp.zeros_like(v)

    a = (F_tot / p.mass) + noise
    x_next, v_next = _semi_implicit_euler(x, v, a, p)
    return (x_next, v_next), (x_next, v_next)


def _init_positions_vels(key, num_left: int, num_right: int, p: SimParams):
    N = num_left + num_right
    key, kL, kR, kjL, kjR, ks1, ks2 = jax.random.split(key, 7)

    yL = jax.random.uniform(kL, (num_left,), minval=p.y_min, maxval=p.y_max)
    yR = jax.random.uniform(kR, (num_right,), minval=p.y_min, maxval=p.y_max)

    jL = jax.random.normal(kjL, (num_left, 2)) * p.jitter_xy
    jR = jax.random.normal(kjR, (num_right, 2)) * p.jitter_xy

    xL = jnp.stack([jnp.full((num_left,), p.x_left), yL], axis=-1) + jL
    xR = jnp.stack([jnp.full((num_right,), p.x_right), yR], axis=-1) + jR
    x0 = jnp.concatenate([xL, xR], axis=0)           # [N,2]

    v0 = jnp.zeros_like(x0)

    rel = p.v_target_rel_jitter
    speed_left  = p.v_target_mag * (1.0 + rel * (jax.random.uniform(ks1, (num_left,)) - 0.5) * 2.0)
    speed_right = p.v_target_mag * (1.0 + rel * (jax.random.uniform(ks2, (num_right,)) - 0.5) * 2.0)

    v_star_left  = jnp.stack([ speed_left, jnp.zeros_like(speed_left)], axis=-1)   # +x
    v_star_right = jnp.stack([-speed_right, jnp.zeros_like(speed_right)], axis=-1) # -x
    v_star = jnp.concatenate([v_star_left, v_star_right], axis=0)

    return x0, v0, v_star


def generate_two_group_dataset(*, num_left: int, num_right: int, T: int, seed: int = 0, params: Optional[SimParams] = None):
    """
    Returns:
      positions:  [T, N, 2] (jnp)
      velocities: [T, N, 2] (jnp)
    """
    p = params or SimParams()
    assert T >= 2, "Need T >= 2 for next-step targets."

    key = jax.random.key(seed)
    key, k_init, k_scan = jax.random.split(key, 3)

    x0, v0, v_star = _init_positions_vels(k_init, num_left, num_right, p)

    def scan_body(carry, rng):
        return _step(carry, rng, v_star, p)

    keys = jax.random.split(k_scan, T - 1)
    (xT, vT), (xs, vs) = jax.lax.scan(scan_body, (x0, v0), keys)

    positions = jnp.concatenate([x0[None, ...], xs], axis=0)
    velocities = jnp.concatenate([v0[None, ...], vs], axis=0)
    return positions, velocities, p, v_star


# ----------------------------- Export: NPZ + Animation -----------------------------

def save_dataset_npz(path: str, positions: jnp.ndarray, velocities: jnp.ndarray, dt: float):
    np.savez(
        path,
        positions=np.asarray(positions),   # [T,N,2]
        velocities=np.asarray(velocities), # [T,N,2]
        dt=np.asarray(dt),
    )
    print(f"Saved dataset to: {path}")


def render_animation_mp4_or_gif(
    positions: jnp.ndarray,
    *,
    num_left: int,
    num_right: int,
    dt: float,
    out_path_mp4: str = "pedestrians.mp4",
    out_path_gif: str = "pedestrians.gif",
    fps: int = 30,
    tail_len: int = 25,     # number of past frames to show as trails
    s: int = 30,            # scatter point size
):
    """
    Try to render an MP4 (ffmpeg). If ffmpeg is unavailable, fall back to GIF (Pillow).
    """
    pos = np.asarray(positions)          # [T,N,2] as numpy
    T, N, _ = pos.shape
    assert N == num_left + num_right

    # Axis limits with a small margin
    xmin, xmax = pos[...,0].min(), pos[...,0].max()
    ymin, ymax = pos[...,1].min(), pos[...,1].max()
    margin_x = 0.1 * (xmax - xmin + 1e-6)
    margin_y = 0.1 * (ymax - ymin + 1e-6)
    xlim = (xmin - margin_x, xmax + margin_x)
    ylim = (ymin - margin_y, ymax + margin_y)

    # Colors: left group (0..num_left-1) vs right group
    colors = np.array(["tab:blue"] * num_left + ["tab:orange"] * num_right)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Two-Group Pedestrian Flow")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Scatter for current positions
    scat = ax.scatter([], [], s=s)

    # Optional trails: a set of Line2D objects per agent
    lines = [ax.plot([], [], lw=1, alpha=0.6, color=colors[i])[0] for i in range(N)]

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Left group', markerfacecolor='tab:blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Right group', markerfacecolor='tab:orange', markersize=8),
    ]
    ax.legend(handles=legend_elems, loc="upper right")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        for ln in lines:
            ln.set_data([], [])
        return [scat, *lines]

    def update(frame):
        # Current positions
        XY = pos[frame]  # [N,2]
        scat.set_offsets(XY)
        scat.set_color(colors)

        # Trails: show last `tail_len` positions (clamped at 0)
        t0 = max(0, frame - tail_len)
        trail = pos[t0:frame+1]  # [L,N,2]
        for i in range(N):
            lines[i].set_data(trail[:, i, 0], trail[:, i, 1])
        ax.set_title(f"Two-Group Pedestrian Flow  |  t={frame}  (dt={dt:.3f})")
        return [scat, *lines]

    interval_ms = int(1000 / fps)
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=T, interval=interval_ms, blit=True)

    # Try MP4 first (ffmpeg), else fallback to GIF (Pillow)
    saved = False
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(out_path_mp4, writer=writer)
        print(f"Saved animation to: {out_path_mp4}")
        saved = True
    except Exception as e:
        print(f"FFmpeg not available or failed ({e}). Falling back to GIF.")
    if not saved:
        try:
            from matplotlib.animation import PillowWriter
            anim.save(out_path_gif, writer=PillowWriter(fps=fps))
            print(f"Saved animation (GIF) to: {out_path_gif}")
            saved = True
        except Exception as e:
            print("Failed to save animation as GIF too:", e)

    plt.close(fig)


# ----------------------------- Example: generate + save -----------------------------

if __name__ == "__main__":
    # Params and sizes
    params = SimParams(
        A=8.0, B=1.2, d0=0.7, r_cut=0.0,
        tau=0.5, sigma=0.0,
        dt=0.05, mass=1.0,
        x_left=-10.0, x_right=10.0,
        y_min=-5.0, y_max=5.0,
        jitter_xy=5.0,
        v_target_mag=1.4, v_target_rel_jitter=0.15
    )
    T = 300
    num_left, num_right = 100, 100
    seed = 123

    # Generate
    positions, velocities, p, v_star = generate_two_group_dataset(
        num_left=num_left,
        num_right=num_right,
        T=T,
        seed=seed,
        params=params,
    )

    # Save NPZ
    save_dataset_npz("pedestrians.npz", positions, velocities, dt=p.dt)
    jnp.save("v_star.npy", v_star)
    # Render animation (MP4 if ffmpeg, else GIF)
    render_animation_mp4_or_gif(
        positions,
        num_left=num_left,
        num_right=num_right,
        dt=p.dt,
        out_path_mp4="pedestrians.mp4",
        out_path_gif="pedestrians.gif",
        fps=30,
        tail_len=30,
        s=28,
    )
