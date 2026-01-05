# IMPORTS

import numpy as onp
import jax.numpy as np
import jax
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

from .utils import wall_energy, align_fn, ttc_potential_fn, ttc_tot, ttc_force, normalize_cap, goal_velocity_force, normal
from .vision import bearing_angle_tot


# WALL INTERACTIONS

def wall_energy_tot(poss, wall, radius, displacement) -> float:
   """
   Compute the total interaction energy between pedestrians and straight walls.
   Walls are modeled to have an interaction energy scaling with 1/r^3.

   Args:
      poss (Array): array of all pedestrians positions
      wall (StraightWall): wall object
      radius (float): radius of pedestrian
      displacement (function): displacement_fn as returned by jax_md.space

   Returns:
      A float value of the total interaction energy between pedestrians and straight walls
   """
   return np.sum(vmap(wall_energy, (0, None, None, None))(poss, wall, radius, displacement))

# CHIRAL INTERACTIONS

def align_tot(R, theta, displacement):
   # only used for chiral AM simuls
   # Alignment factor
   align = vmap(vmap(align_fn, (0, None, 0)), (0, 0, None))

   # Displacement between all points
   dR = space.map_product(displacement)(R, R)

   return np.sum(align(dR, theta, theta), axis=1)

# PEDESTRIAN INTERACTIONS

def ttc_potential_tot(pos, V, R, displacement, k=1.5, t_0=3.0) -> jax.Array[float]:
   """
   The potential energy of pedestrian interaction, according to
   the anticipatory interaction law detailed in [1].

   Args:
      pos (ndarray)     : position vector of all particles
      V (ndarray)       : velocity vector "   "      "
      R (float)         : collision radius of a particle
      displacement (fn) : displacement function produced by jax_md.space
      k, t_0 (floats)   : interaction params

   Returns:
      Array of potential energy of each particle

   [1] I. Karamouzas, B. Skinner, Stephen J. Guy.
   "Universal Power Law Governing Pedestrian Interactions"
   """
   # should use force_fn directly instead of potential energy
   return np.array(np.sum(smap._diagonal_mask(ttc_potential_fn(k, ttc_tot(pos, V, R, displacement), t_0))) / 2)

def ttc_force_unsummed_tot(pos, V, R, displacement, k=1.5, t_0=3.0) -> jax.Array[jax.Array[jax.Array[float]]]:
   """
   The pedestrian social interaction between all pairs of pedestrians, according to
   the anticipatory interaction law detailed in [1].

   Arguments:
      pos (ndarray): position vector of all particles, shape (N, 2)
      V (ndarray): velocity vector of all particles, shape (N, 2)
      R (float): collision radius of a particle
      displacement (fn): displacement function produced by jax_md.space
      k (float): interaction strength param
      t_0 (float): characteristic interaction time

   Returns:
      Array of shape (N, N, 2), where array[i, j] denotes the force experienced by i due to j.

   [1] I. Karamouzas, B. Skinner, Stephen J. Guy.
   "Universal Power Law Governing Pedestrian Interactions"
   """
   force_fn = vmap(vmap(ttc_force, (0, 0, None, None, None, None)), (0, None, 0, None, None, None))

   dpos = space.map_product(displacement)(pos, pos)

   return normalize_cap(force_fn(dpos, V, V, R, k, t_0), 5)

def ttc_force_tot(pos, V, R, displacement, k=1.5, t_0=3.0) -> jax.Array[jax.Array[float]]:
   """
   The pedestrian social force experienced by each individual pedestrian, according to
   the anticipatory interaction law detailed in [1].

   Arguments:
      pos (ndarray): position vector of all particles, shape (N, 2)
      V (ndarray): velocity vector of all particles, shape (N, 2)
      R (float): collision radius of a particle
      displacement (fn): displacement function produced by jax_md.space
      k (float): interaction strength param. Defaults to 1.5
      t_0 (float): characteristic interaction time. Defaults to 3.0

   Returns:
      Array of shape (N, 2), where array[i] denotes the total force experienced by i.

   [1] I. Karamouzas, B. Skinner, Stephen J. Guy.
   "Universal Power Law Governing Pedestrian Interactions"
   """
   return np.sum(ttc_force_unsummed_tot(pos, V, R, displacement, k, t_0), axis=1)

def ttc_visual_force_unsummed_tot(pos, V, R, displacement, visual_action, k=1.5, t_0=3.0) -> jax.Array[jax.Array[jax.Array[float]]]:
   """
   The pedestrian social force according to the anticipatory interaction
   law detailed in [1], augmented with visual information of each pedestrian.

   Arguments:
      pos (ndarray): position vector of all particles, shape (N, 2)
      V (ndarray): velocity vector of all particles, shape (N, 2)
      R (float): collision radius of a particle
      displacement (fn): displacement function produced by jax_md.space
      visual_action (fn): a function that takes in a bearing angle matrix of shape (N, N, 3, 1) and returns a matrix of size (N, N, 1) whose entries are numbers in [0, 1]
      k (float): interaction strength param
      t_0 (float): characteristic interaction time

   Returns:
      Array of shape (N, N, 2), where array[i, j] denotes the force experienced by i due to j.

   [1] I. Karamouzas, B. Skinner, Stephen J. Guy.
   "Universal Power Law Governing Pedestrian Interactions"
   """
   return visual_action(bearing_angle_tot(pos, V, R, displacement)) * ttc_force_unsummed_tot(pos, V, R, displacement, k, t_0)

def ttc_visual_force_tot(pos, V, R, displacement, visual_action, k=1.5, t_0=3.0) -> jax.Array[jax.Array[float]]:
   """
   The pedestrian social force according to the anticipatory interaction
   law detailed in [1], augmented with visual information of each pedestrian.

   Arguments:
      pos (ndarray): position vector of all particles, shape (N, 2)
      V (ndarray): velocity vector of all particles, shape (N, 2)
      R (float): collision radius of a particle
      displacement (fn): displacement function produced by jax_md.space
      visual_action (fn): a function that takes in a bearing angle matrix of shape (N, N, 3, 1) and returns a matrix of size (N, N, 1) whose entries are numbers in [0, 1]
      k (float): interaction strength param. Defaults to 1.5
      t_0 (float): characteristic interaction time. Defaults to 3.0

   Returns:
      Array of shape (N, 2), where array[i] denotes the total force experienced by i.

   [1] I. Karamouzas, B. Skinner, Stephen J. Guy.
   "Universal Power Law Governing Pedestrian Interactions"
   """
   return np.sum(ttc_visual_force_unsummed_tot(pos, V, R, displacement, visual_action, k, t_0), axis=1)

def goal_velocity_force_tot(velocities, goal_speeds, goal_orientations=None) -> jax.Array[jax.Array[float]]:
   """
   Force representing pedestrian's target velocity.

   Arguments:
      velocities (jax.Array): array of particles' velocities
      goal_speeds (jax.Array): array of particles' preferred speeds
      goal_orientations (jax.Array | None): array of particles' preferred directions. If None is provided, then it is assumed that there is no preferred direction.

   Returns:
      jax.Array of force acting on each particle
   """
   if goal_orientations is None:
      return (normal(goal_speeds, np.atan2(velocities[:, 1], velocities[:, 0])) - velocities) / .5
   return vmap(goal_velocity_force, (0, 0, 0))(velocities, goal_speeds, goal_orientations)
