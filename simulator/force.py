# IMPORTS

import numpy as onp
import jax.numpy as np
from jax import random
from jax import jit
from jax import vmap
from jax import lax
from jax import config
config.update("jax_enable_x64", True)

from jax_md import space, smap
from jax_md.util import f32

from collections import namedtuple

vectorize = np.vectorize

from functools import partial

from .basis import LegendrePolynomial, LaguerrePolynomial, LaguerreBase, LegendreBase
from .utils import wall_energy, align_fn, ttc_potential_fn, ttc_tot, ttc_force, normalize_cap, goal_velocity_force

# WALL INTERACTIONS

def wall_energy_tot(poss, wall, radius, displacement):
   return np.sum(vmap(wall_energy, (0, None, None, None))(poss, wall, radius, displacement))

# CHIRAL INTERACTIONS

def align_tot(R, theta, displacement):
   # Alignment factor
   align = vmap(vmap(align_fn, (0, None, 0)), (0, 0, None))

   # Displacement between all points
   dR = space.map_product(displacement)(R, R)

   return np.sum(align(dR, theta, theta), axis=1)

# PEDESTRIAN INTERACTIONS

def ttc_potential_tot(pos, V, R, displacement, k=1.5, t_0=3.0):
   """
   The potential energy of pedestrian interaction, according to
   the anticipatory interaction law detailed in [1].

   Inputs:
      pos (ndarray)     : position vector of all particles
      V (ndarray)       : velocity vector "   "      "
      R (float)         : collision radius of a particle
      displacement (fn) : displacement function produced by jax_md.space
      k, t_0 (floats)   : interaction params

   Output:
      ndarray of potential energy of each particle

   [1] I. Karamouzas, B. Skinner, Stephen J. Guy.
   "Universal Power Law Governing Pedestrian Interactions"
   """
   return np.array(np.sum(smap._diagonal_mask(ttc_potential_fn(k, ttc_tot(pos, V, R, displacement), t_0))) / 2)

def ttc_force_tot(pos, V, R, displacement, k=1.5, t_0=3.0):
   force_fn = vmap(vmap(ttc_force, (0, 0, None, None, None, None)), (0, None, 0, None, None, None))

   dpos = space.map_product(displacement)(pos, pos)

   return np.sum(normalize_cap(force_fn(dpos, V, V, R, k, t_0), 5), axis=1)

def goal_velocity_force_tot(velocities, goal_speeds, goal_orientations=None):
   """
   Force representing pedestrian's target velocity.

   Arguments:
      velocities (jax.Array): array of particles' velocities
      goal_speeds (jax.Array): array of particles' preferred speeds
      goal_orientations (jax.Array / None): array of particles' preferred directions. If
      None is provided, then it is assumed that there is no preferred direction.

   Output:
      jax.Array of force acting on each particle
   """
   if goal_orientations is None:
      return (goal_speeds / np.linalg.norm(velocities, axis=1) - 1) * velocities / .5
   return vmap(goal_velocity_force, (0, 0, 0))(velocities, goal_speeds, goal_orientations)

# BASIS POLYNOMIAL EXPANSION

def general_force_generator(weight_paral_arr, weight_perpen_arr, v_0, d_0):
   # weight tensor has shape (mu_0, mu_1, mu_2)

   def general_force_magnitude(scaled_v, scaled_pos, proj, weight_arr):
      I, J, K = weight_arr.shape
      lag_i = np.arange(0, I)
      lag_j = np.arange(0, J)
      leg_k = np.arange(0, K)

      def updater(arr, idx, poly_type, val):
         # True == Laguerre
         # False == Legendre
         new_arr = np.where(poly_type,
                            arr.at[idx].set(LaguerreBase(idx)(val)),
                            arr.at[idx].set(LegendreBase(idx)(val)))
         return new_arr, idx

      lag_i = lax.scan(partial(updater, poly_type=True, val=scaled_v), lag_i, lag_i)[0]
      lag_j = lax.scan(partial(updater, poly_type=True, val=scaled_pos), lag_j, lag_j)[0]
      leg_k = lax.scan(partial(updater, poly_type=False, val=proj), leg_k, leg_k)[0]

      expansion = weight_arr * np.tensordot(np.tensordot(lag_i, lag_j, axes=0), leg_k, axes=0)

      return np.sum(expansion) * np.exp(-(scaled_v + scaled_pos)/2)

   def general_force(dpos, V_i, V_j):
      dv = V_i - V_j

      n_pos =  np.linalg.norm(dpos)
      n_v = np.linalg.norm(dv)

      unit_pos = dpos / n_pos
      unit_v = dv / n_v

      scaled_pos = n_pos / d_0
      scaled_v = n_v / v_0
      proj = np.dot(dv, dpos) / (scaled_pos * scaled_v)

      # force calc
      force = (general_force_magnitude(scaled_v, scaled_pos, proj, weight_paral_arr) * unit_v +
              general_force_magnitude(scaled_v, scaled_pos, proj, weight_perpen_arr) * np.matmul(np.identity(2) - np.matmul(unit_v, np.transpose(unit_v)), unit_pos))

      return force

   return general_force
