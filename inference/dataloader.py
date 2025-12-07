# Assumes positions/velocities are shaped [T, N, 2].
from typing import Dict, Iterator, Optional
import jax
import jax.numpy as jnp
import jax.random as jr

def tp_to_linear(t: int, i: int, N: int) -> int:
    """Map (t, i) -> k = t*N + i."""
    return int(t) * int(N) + int(i)

def linear_to_tp(k: int, N: int):
    """Inverse mapping: (t, i) = (k // N, k % N)."""
    return divmod(int(k), int(N))

def others_indices(i: jnp.ndarray, N: int) -> jnp.ndarray:
    """
    Return integer indices of all people except i.
    Works under vmap/JIT (no boolean indexing).
    """
    idx = jnp.arange(N)            # [N]
    # swap idx[i] <-> idx[N-1] to move i to the end
    idx = idx.at[i].set(idx[-1])
    idx = idx.at[-1].set(i)
    return idx[:-1]                # [N-1]

def gather_others(x_t: jnp.ndarray, i: jnp.ndarray) -> jnp.ndarray:
    """
    x_t: [N, ...] (e.g., [N, 2])
    returns x_t without row i, shape [N-1, ...]
    """
    idx_others = others_indices(i, x_t.shape[0])
    return jnp.take(x_t, idx_others, axis=0)


# ----------------------------- Random-access dataset -----------------------------

class PeopleTrajNextDatasetJAX:
    """
    Dataset over (time / dT, person) items for trajectory velocity prediction.

    Returns a dict with:
      pos           [T, 2]        (pos[t : t+dT, i])
      vel           [T, 2]        (vel[t : t+dT, i])
      person_index  [] (int32) (i)
      time_index    [] (int32) (t)
      others_pos    [T, N-1, 2]   (pos[t : t+dT, !=i])
      others_vel    [T, N-1, 2]   (vel[t : t+dT, !=i])
    """
    def __init__(self, positions, velocities):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, k: int) -> Dict[str, jnp.ndarray]:
        pass


class PeopleTPNextDatasetJAX:
    """
    Dataset over (time, person) items for next-step velocity prediction.

    Returns a dict with:
      pos           [2]        (pos[t, i])
      vel           [2]        (vel[t, i])
      next_vel      [2]        (vel[t+1, i])  <-- target
      person_index  [] (int32) (i)
      time_index    [] (int32) (t)
      others_pos    [N-1, 2]   (pos[t, !=i])
      others_vel    [N-1, 2]   (vel[t, !=i])
    """
    def __init__(self, positions, velocities):
        pos = jnp.asarray(positions)
        vel = jnp.asarray(velocities)
        assert pos.shape == vel.shape, "positions and velocities must match"
        assert pos.ndim == 3 and pos.shape[-1] == 2, "Expected [T, N, 2]"
        T, N, _ = pos.shape
        assert T >= 2, "Need at least 2 timepoints for next-step target"

        self.pos = pos
        self.vel = vel
        self.T = T
        self.N = N

    def __len__(self) -> int:
        return (self.T - 1) * self.N  # exclude t = T-1

    def __getitem__(self, k: int) -> Dict[str, jnp.ndarray]:
        t, i = divmod(int(k), int(self.N))  # Python ints -> OK outside JIT
        pos_t = self.pos[t]                 # [N, 2]
        vel_t = self.vel[t]                 # [N, 2]

        # Use integer gather to avoid boolean indexing
        others_pos = gather_others(pos_t, jnp.int32(i))   # [N-1, 2]
        others_vel = gather_others(vel_t, jnp.int32(i))   # [N-1, 2]

        return {
            "pos":          pos_t[i],             # [2]
            "vel":          vel_t[i],             # [2]
            "next_vel":     self.vel[t + 1, i],   # [2]
            "person_index": jnp.int32(i),
            "time_index":   jnp.int32(t),
            "others_pos":   others_pos,           # [N-1, 2]
            "others_vel":   others_vel,           # [N-1, 2]
        }


# ----------------------------- Batched builder (vectorized) -----------------------------

def make_batch_fn_traj_next(positions: jnp.ndarray, velocities: jnp.ndarray, timeframe: jnp.int32):
    """
    Returns batch_from_indices(idxs, time_idxs), where idxs are linear indices in [0, (T-1)*N / T_traj).

    Output shapes (T = traj_time, B = len(idxs)):
      pos [B, T_traj, 2], vel [B, T_traj, 2],
      person_index [B, T_traj], time_index [B, T_traj],
      others_pos [B, T_traj, N-1, 2], others_vel [B, T_traj, N-1, 2]
    """
    pos = jnp.asarray(positions)
    vel = jnp.asarray(velocities)
    T, N, _ = pos.shape
    assert T >= 2, "Need at least 2 timepoints"
    assert timeframe >= 2, "Need at least 2 frames"

    def _one(idx: jnp.int32):
        # Split linear -> (t, i) with JAX-native divmod (works under vmap/JIT)
        t, i = jnp.divmod(idx, N)          # scalars (JAX arrays)
        pos_t = pos[t]                     # [N, 2]
        vel_t = vel[t]                     # [N, 2]

        # Integer-gather "others" (no boolean mask)
        others_pos = gather_others(pos_t, i)   # [N-1, 2]
        others_vel = gather_others(vel_t, i)   # [N-1, 2]

        return (
            pos_t[i],             # [2]
            vel_t[i],             # [2]
            i,                    # person_index []
            t,                    # time_index []
            others_pos,           # [N-1, 2]
            others_vel,           # [N-1, 2]
        )

    def _one_traj(idx: jnp.int32):
        idxs = jnp.linspace(idx, idx + (timeframe - 1) * N, timeframe)

        pos_traj, vel_traj, i_traj, t_traj, others_traj, others_vel = jax.vmap(_one)(idxs.astype(jnp.int32))
        return (
            pos_traj,             # [T_traj, 2]
            vel_traj,             # [T_traj, 2]
            i_traj[0],            # person_index []
            t_traj[0],            # start_time_index []
            others_traj,          # [T_traj, N-1, 2]
            others_vel,           # [T_traj, N-1, 2]
        )


    def batch_from_indices(idxs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        pos_b, vel_b, i_b, t_b, o_pos_b, o_vel_b = jax.vmap(_one_traj)(idxs.astype(jnp.int32))
        return {
            "pos":          pos_b,
            "vel":          vel_b,
            "person_index": i_b,
            "time_index":   t_b,
            "others_pos":   o_pos_b,
            "others_vel":   o_vel_b,
        }

    return batch_from_indices


def make_batch_fn_next(positions: jnp.ndarray, velocities: jnp.ndarray):
    """
    Returns batch_from_indices(idxs), where idxs are linear indices in [0, (T-1)*N).

    Output shapes (B = len(idxs)):
      pos [B, 2], vel [B, 2], next_vel [B, 2],
      person_index [B], time_index [B],
      others_pos [B, N-1, 2], others_vel [B, N-1, 2]
    """
    pos = jnp.asarray(positions)
    vel = jnp.asarray(velocities)
    T, N, _ = pos.shape
    assert T >= 2, "Need at least 2 timepoints"

    def _one(idx: jnp.int32):
        # Split linear -> (t, i) with JAX-native divmod (works under vmap/JIT)
        t, i = jnp.divmod(idx, N)          # scalars (JAX arrays)
        pos_t = pos[t]                     # [N, 2]
        vel_t = vel[t]                     # [N, 2]

        # Integer-gather "others" (no boolean mask)
        others_pos = gather_others(pos_t, i)   # [N-1, 2]
        others_vel = gather_others(vel_t, i)   # [N-1, 2]

        return (
            pos_t[i],             # [2]
            vel_t[i],             # [2]
            vel[t + 1, i],        # next_vel [2]
            i,                    # person_index []
            t,                    # time_index []
            others_pos,           # [N-1, 2]
            others_vel,           # [N-1, 2]
        )

    def batch_from_indices(idxs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        pos_b, vel_b, nvel_b, i_b, t_b, o_pos_b, o_vel_b = jax.vmap(_one)(idxs.astype(jnp.int32))
        return {
            "pos":          pos_b,
            "vel":          vel_b,
            "next_vel":     nvel_b,
            "person_index": i_b,
            "time_index":   t_b,
            "others_pos":   o_pos_b,
            "others_vel":   o_vel_b,
        }

    return batch_from_indices


# ----------------------------- Simple epoch data loader -----------------------------

def data_loader_traj_next(
    positions,
    velocities,
    batch_size: int,
    timeframe: int,
    rng_key: jax.Array,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Iterator[Dict[str, jnp.ndarray]]:
    """
    Iterate once over all valid (t, i) with t in [0..T-2] (total = (T-1)*N).
    Yields dict batches with shapes documented above.
    """
    # edit: total, idxs,
    pos = jnp.asarray(positions)
    vel = jnp.asarray(velocities)
    T, N, _ = pos.shape
    batch_from_indices = make_batch_fn_traj_next(pos, vel, timeframe)

    all_idxs = jnp.arange(T * N, dtype=jnp.int32)  # k = t*N + i
    idxs = all_idxs[(all_idxs // N) % timeframe == 0] # all start time indices
    total = idxs.shape[0]

    perm = jr.permutation(rng_key, idxs) if shuffle else idxs

    limit = (total // batch_size) * batch_size if drop_last else total
    for start in range(0, limit, batch_size):
        batch_idxs = perm[start : start + batch_size]
        yield batch_from_indices(batch_idxs)



def data_loader_tp_next(
    positions,
    velocities,
    batch_size: int,
    rng_key: jax.Array,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Iterator[Dict[str, jnp.ndarray]]:
    """
    Iterate once over all valid (t, i) with t in [0..T-2] (total = (T-1)*N).
    Yields dict batches with shapes documented above.
    """
    pos = jnp.asarray(positions)
    vel = jnp.asarray(velocities)
    T, N, _ = pos.shape
    total = (T - 1) * N
    batch_from_indices = make_batch_fn_next(pos, vel)

    idxs = jnp.arange(total, dtype=jnp.int32)  # k = t*N + i
    perm = jr.permutation(rng_key, total) if shuffle else idxs

    limit = (total // batch_size) * batch_size if drop_last else total
    for start in range(0, limit, batch_size):
        batch_idxs = perm[start : start + batch_size]
        yield batch_from_indices(batch_idxs)
