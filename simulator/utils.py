# IMPORTS

import numpy as onp
import jax.numpy as np
from jax import random
from jax import jit
from jax import vmap
from jax import lax
from jax import config
config.update("jax_enable_x64", True)

from jax_md import space, smap, energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

from collections import namedtuple

vectorize = np.vectorize

from functools import partial

from .basis import LegendrePolynomial, LaguerrePolynomial, LegendreBase, LaguerreBase

# from typing import Union, Callable

# SPACES

@jit
def angle_correct(theta):
    return np.where(
        theta < 0,
        theta + 2 * onp.pi,
        np.where(theta > 2 * onp.pi, theta - 2 * onp.pi, theta),
    )

def _normalize(v, v_lim=0):
    v_norm = np.linalg.norm(v, keepdims=True)
    return np.where(v_norm > v_lim, v/v_norm, v)

@jit
def normalize_cap(v, v_lim=0):
   return vmap(_normalize, (0, None))(v, v_lim)

# @partial(jit, static_argnums=1)
def wall_dist(pos, start, end, displacement):
   """
   Computes displacement from a particle to a wall. A wall is parameterized by
   its two endpoints.

   Inputs:
      pos (Array): array denoting particle position
      start, end (Array): arrays denoting wall positions
      displacement (fn): function to compute distance between 2 points

   Output:
      Array denoting displacement from particle to wall
   """
   wall_len = np.dot(end - start, end - start)
   # t = max(0, min(1, np.dot(pos - start, end - start) / wall_len))
   t = np.max(np.array([0, np.min(np.array([1, np.dot(pos - start, end - start) / wall_len]), axis = 0)]), axis = 0)
   proj = start + t * (end - start)
   return pos - proj

# @partial(jit, static_argnums=1)
def wall_energy(pos, wall, radius, displacement):
   """
   Used to model the repulsion between a particle and a wall.

   Inputs:
      pos (Array): particle position
      wall (Wall obj): object representing a wall, with start and end attrs

   Output:
      Interaction between particle and the wall
   """
   start = lax.stop_gradient(wall.start)
   end = lax.stop_gradient(wall.end)

   dist = wall_dist(pos, start, end, displacement)

   # return 5 * radius / (np.linalg.norm(dist) - radius)
   return 1 / (3 * (np.linalg.norm(dist) / radius) ** 3)

# CHIRAL PARTICLES

@vmap
def normal(v, theta):
  return np.array([v * np.cos(theta), v * np.sin(theta)])

@jit
def align_fn(dr, theta_i, theta_j):
   align_spd = np.sin(theta_j - theta_i)
   dR = space.distance(dr)
   return np.where(dR < 1., align_spd, 0.)

# PEDESTRIANS

def init_velocity(key, speed_mean, N):
    """
    Initializes particles with 2D Maxwell-Boltzmann distrib, with given mean speed
    """
    return np.sqrt(2/ onp.pi) * speed_mean * random.normal(key, shape=(N, 2))

def init_goal_speed(key, N, speed_mean=1.3, speed_var=0.3):
    """
    Initializes pedestrians with their own preferred walking speeds.
    The preferred speed is normally distributed around speed_mean, with
    s.d speed_var, and truncated to fall into [speed_mean - speed_var, speed_mean + speed_var]
    """
    return speed_mean + speed_var * random.truncated_normal(key,
                                                            lower=-1,
                                                            upper=+1,
                                                            shape=(N,))

@jit
def ttc_potential_fn(k, t, t_0):
   return np.array((k / np.square(t)) * np.exp(-t/ t_0))


@jit
def time_to_collide(dpos, V_i, V_j, R):
   """
   dpos = pos_i - pos_j, V_i, V_j : ndarrays
   R : float

   The time to collide is the smaller root of the
   following quadratic equation at^2 + bt + c = 0
   """
   # stop grad for non-positional args
   V_i = lax.stop_gradient(V_i)
   V_j = lax.stop_gradient(V_j)
   R = lax.stop_gradient(R)

   t_max = 99
   dv = V_i - V_j

   a = np.dot(dv, dv)
   b = - np.dot(dpos, dv)
   c = np.dot(dpos, dpos) - np.square(2 * R)

   det = np.square(b) - a * c

   t1 = np.where(det > 0, np.nan_to_num((-b - np.sqrt(det))/(a)), t_max)
   t2 = np.where(det > 0, np.nan_to_num((-b + np.sqrt(det))/(a)) , t_max)

   t1 = np.where(t1 < 0, np.where(t2 > 0, 0.1, t_max), t1)  # maybe should be 0?

   t1 = np.where(t1 > t_max, t_max, t1)

   return t1

@jit
def ttc_force(dpos, V_i, V_j, R, k, t_0):
   t_max = 99
   dv = V_i - V_j

   sq_dist = np.dot(dpos, dpos)
   sq_rad = np.square(2 * R)
   adjusted_sq_rad = np.where(sq_rad > sq_dist, .99 * sq_dist, sq_rad)

   a = np.dot(dv, dv)
   b = - np.dot(dpos, dv)
   c = sq_dist - adjusted_sq_rad
   # c = np.dot(dpos, dpos) - np.square(2 * R)

   det = np.square(b) - a * c

   t = (b - np.sqrt(det)) / a

   return np.where(np.logical_or(det < 0, np.logical_and(a < 0.001, a > -0.001)),
                   np.array([0., 0.]),
                   np.where(np.logical_or(t < 0, t > t_max),
                            np.array([0., 0.]),
                            - k*np.exp(-t/t_0)*(dv - (dv * b - dpos * a)/(np.sqrt(det)))/(a*np.square(t))*(2/t+ 1/t_0)))

@jit
def safe_time_to_collide(dpos, V_i, V_j, R, i, j):
   return np.where(i != j, time_to_collide(dpos, V_i, V_j, R), 99)

def ttc_tot(pos, V, R, displacement):
   ttc = vmap(vmap(
      safe_time_to_collide, (0, None, 0, None, None, 0)
      ), (0, 0, None, None, 0, None)
      )
   dpos = space.map_product(displacement)(pos, pos)

   dim = V.shape[0]

   I = np.arange(0, dim)
   J = np.arange(0, dim)

   return ttc(dpos, V, V, R, I, J)

def _ttc_force_tot(pos, V, R, displacement, k=1.5, t_0=3.0):
   force_fn = vmap(vmap(ttc_force, (0, 0, None, None, None, None)), (0, None, 0, None, None, None))

   dpos = space.map_product(displacement)(pos, pos)

   return np.sum(force_fn(dpos, V, V, R, k, t_0), axis=1)

# def goal_velocity_force(state):
#    if state.goal_orientation is None:
#       return (normal(state.goal_speed, state.orientation()) - state.velocity) / .5
#    return (normal(state.goal_speed, state.goal_orientation) - state.velocity) / .5

def goal_velocity_force(velocity, goal_speed, goal_orientation):
   return (goal_speed * np.array([np.cos(goal_orientation), np.sin(goal_orientation)]) - velocity) / .5

def numpify_wall(wall):
   return np.array([wall.start, wall.end])

# TEST SECTION

import time

# BASIS POLYNOMIAL EXPANSION

# Issues: depend on onp funcs, for loops => not compatible w/ JAX, but can be fixed.
# Requires creating ijk Polynomial objs => inefficient?
def general_force_generator_TEST_1(weight_paral_arr, weight_perpen_arr, v_0, d_0):
   init_start_time = time.time()
   # weight tensor has shape (mu_0, mu_1, mu_2)
   def term_generator(i, j, k):
      lag_i = LaguerrePolynomial(np.array([0] * (i - 1) + [1]))
      lag_j = LaguerrePolynomial(np.array([0] * (j - 1) + [1]))
      leg_k = LegendrePolynomial(np.array([0] * (k - 1) + [1]))

      def term(scaled_v, scaled_pos, proj):
         return lag_i(scaled_v) * lag_j(scaled_pos) * leg_k(proj) * np.exp(-(scaled_v + scaled_pos)/2)

      return term

   def general_force_magnitude(scaled_v, scaled_pos, proj, weight_arr):
      # NOT COMPATIBLE WITH JAX (FOR NOW)
      # operation to sum the terms while multiplied over the weights
      magnitude = 0.
      for idw, weight in onp.ndenumerate(weight_arr):
         i, j, k = idw
         magnitude += weight * term_generator(i, j, k)(scaled_v, scaled_pos, proj)
      return magnitude

   def general_force(dpos, V_i, V_j):
      start_time = time.time()

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

      end_time = time.time()
      print(f"test_1_runtime = {end_time - start_time}")

      return force

   init_end_time = time.time()
   print(f"test_1_init_runtime = {init_end_time - init_start_time}")

   return general_force

# Issues: Use tuple of funcs to reduce computation cost => Indexing into tuple is not supported by jitted funcs!
# Computation cost not significantly reduced => lol im js gnna use the above fn
def general_force_generator_TEST_2(weight_paral_arr, weight_perpen_arr, v_0, d_0):
   init_start_time = time.time()

   # arrs are assumed to have shape (i, j, k)
   # identify maximum degrees of each basis fn
   i, j, k = np.max(np.array((weight_paral_arr.shape, weight_perpen_arr.shape)), axis=0)
   ij = np.max(np.array([i, j]))

   # generate enough basis functions for use:
   # PAST_ERROR: unsupported operand type(s) for *: 'list' and 'ArrayImpl'
   # laguerres = [LaguerrePolynomial(np.array([0] * l + [1])) for l in np.arange(0, ij + 1)]
   laguerres = tuple([LaguerrePolynomial(np.array([0] * l + [1])) for l in range(0, ij + 1)])
   legendres = tuple([LegendrePolynomial(np.array([0] * l + [1])) for l in range(0, k + 1)])

   # unroll arrays
   paral_shape = np.array(weight_paral_arr.shape)
   perpen_shape = np.array(weight_perpen_arr.shape)
   weights_paral = weight_paral_arr.flatten()
   weights_perpen = weight_perpen_arr.flatten()

   # # indexer for flattened array:
   # def indexer(idx, paral):
   #    a, b, c = np.where(paral, paral_shape, perpen_shape)
   #    u2 = idx // (a * b)
   #    u = idx % (a * b)
   #    u1 = u // a
   #    u0 = u % a
   #    return u0, u1, u2

   # # operation to sum the terms while multiplied over the weights
   # # ERROR: TracerIntegerConversionError
   # # PROPOSED_FIX: make laguerres and legendres into static arguments
   # @partial(jit, static_argnames=['laguerres', 'legendres', 'idx'])
   # def full_term(idx, magn, paral, scaled_v, scaled_pos, proj, laguerres=laguerres, legendres=legendres):
   #    u0, u1, u2 = indexer(idx, paral)
   #    magn += laguerres[u0](scaled_v) * laguerres[u1](scaled_pos) * legendres[u2](proj) * np.exp(-(scaled_v + scaled_pos)/2)
   #    return magn

   # @partial(jit, static_argnames=['paral'])
   def general_force_magnitude(scaled_v, scaled_pos, proj, weights_arr, paral):
      #weights_arr is the flattened weight array, with shape (size,)
      #indexer for flattened array:
      def indexer(idx):
         a, b, c = np.where(paral, paral_shape, perpen_shape)
         u2 = idx // (a * b)
         u = idx % (a * b)
         u1 = u // a
         u0 = u % a
         return u0, u1, u2

      #operation to sum the terms while multiplied over the weights
      #ERROR: TracerIntegerConversionError
      #PROPOSED_FIX: make laguerres and legendres into static arguments
      def term(idx, magn):
         u0, u1, u2 = indexer(idx)
         magn += weights_arr[idx] * laguerres[u0](scaled_v) * laguerres[u1](scaled_pos) * legendres[u2](proj) * np.exp(-(scaled_v + scaled_pos)/2)
         return magn

      # term = partial(full_term, paral=paral, scaled_v=scaled_v, scaled_pos=scaled_pos, proj=proj)
      magnitude = 0.
      for idx in range(weights_arr.size):
         magnitude = term(idx, magnitude)
      # magnitude = lax.fori_loop(0, weights_arr.size, term, 0.)
      return magnitude

   def general_force(dpos, V_i, V_j):
      start_time = time.time()

      dv = V_i - V_j

      n_pos =  np.linalg.norm(dpos)
      n_v = np.linalg.norm(dv)

      unit_pos = dpos / n_pos
      unit_v = dv / n_v

      scaled_pos = n_pos / d_0
      scaled_v = n_v / v_0
      proj = np.dot(dv, dpos) / (scaled_pos * scaled_v)

      force = (general_force_magnitude(scaled_v, scaled_pos, proj, weights_paral, True) * unit_v +
              general_force_magnitude(scaled_v, scaled_pos, proj, weights_perpen, False) * np.matmul(np.identity(2) - np.matmul(unit_v, np.transpose(unit_v)), unit_pos))

      end_time = time.time()
      print(f"test_2_runtime = {end_time - start_time}")
      return force

   init_end_time = time.time()
   print(f"test_2_init_runtime = {init_end_time - init_start_time}")

   return general_force

# Issues: Fail during reverse-mode diff because it is using fori_loop instead of scan.
def general_force_generator_TEST_3(weight_paral_arr, weight_perpen_arr, v_0, d_0):
   init_start_time = time.time()
   # weight tensor has shape (mu_0, mu_1, mu_2)

   def general_force_magnitude(scaled_v, scaled_pos, proj, weight_arr):
      # PROPOSED SOLUTION: USE TENSORDOT
      I, J, K = weight_arr.shape
      lag_i = np.zeros(I)
      lag_j = np.zeros(J)
      leg_k = np.zeros(K)

      # ISSUE: idx is a JaxprTracer!
      def updater(idx, arr, poly_type, val):
         # True == Laguerre
         # False == Legendre
         new_arr = np.where(poly_type,
                            arr.at[idx].set(LaguerreBase(idx)(val)),
                            arr.at[idx].set(LegendreBase(idx)(val)))
         return new_arr

      lag_i = lax.fori_loop(0, I, partial(updater, poly_type=True, val=scaled_v), lag_i)
      lag_j = lax.fori_loop(0, J, partial(updater, poly_type=True, val=scaled_pos), lag_j)
      leg_k = lax.fori_loop(0, K, partial(updater, poly_type=False, val=proj), leg_k)

      expansion = weight_arr * np.tensordot(np.tensordot(lag_i, lag_j, axes=0), leg_k, axes=0)

      return np.sum(expansion) * np.exp(-(scaled_v + scaled_pos)/2)

   def general_force(dpos, V_i, V_j):
      start_time = time.time()
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

      end_time = time.time()
      print(f"test_3_runtime = {end_time - start_time}")
      return force

   init_end_time = time.time()
   print(f"test_3_init_runtime = {init_end_time - init_start_time}")
   return general_force


def general_force_generator_TEST_4(weight_paral_arr, weight_perpen_arr, v_0, d_0):
   init_start_time = time.time()
   # weight tensor has shape (mu_0, mu_1, mu_2)

   def general_force_magnitude(scaled_v, scaled_pos, proj, weight_arr):
      # PROPOSED SOLUTION: USE TENSORDOT
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

   paral_force_mag = partial(general_force_magnitude, weight_arr=weight_paral_arr)
   perpen_force_mag = partial(general_force_magnitude, weight_arr=weight_perpen_arr)

   def general_force(dpos, V_i, V_j):
      start_time = time.time()
      dv = V_i - V_j

      n_pos =  np.linalg.norm(dpos)
      n_v = np.linalg.norm(dv)

      unit_pos = dpos / n_pos
      unit_v = dv / n_v

      scaled_pos = n_pos / d_0
      scaled_v = n_v / v_0
      proj = np.dot(dv, dpos) / (scaled_pos * scaled_v)

      # force calc
      force = (paral_force_mag(scaled_v, scaled_pos, proj) * unit_v +
              perpen_force_mag(scaled_v, scaled_pos, proj) * np.matmul(np.identity(2) - np.matmul(unit_v, np.transpose(unit_v)), unit_pos))

      end_time = time.time()
      print(f"test_4_runtime = {end_time - start_time}")
      return force

   init_end_time = time.time()
   print(f"test_4_init_runtime = {init_end_time - init_start_time}")
   return general_force

# DATA STRUCT

class LoopedList(object):
   def __init__(self, elts):
      self.elts = elts

   def __repr__(self):
      return f"LoopedList({self.elts})"

   def __eq__(self, other):
      return self.elts == other.elts

   def __len__(self):
      return len(self.elts)

   def __getitem__(self, idx):
      return self.elts[idx % len(self)]

   def __setitem__(self, idx, val):
      self.elts[idx % len(self)] = val

   def __delitem__(self, idx):
      del self.elts[idx % len(self)]

# CONDITIONAL FUNCTION FOR GOAL CHANGE:
# condition = condition(curr_pos, curr_goal)

def dgoal_generator(transitions, goals):
   """
   Inputs:
      transitions (list(fn: pos, goal -> new_goal)): list of M transitioning functions
      goals (Array(Coords)): array of M goals

   Output:
      func: pos, old_goals => new_goals
   """
   # assert len(transitions) == goals.shape[0], f"There are {len(goals)} goals but {len(transitions)} transitions was provided!"
   M = len(transitions)

   def dgoal(pos, old_goals):
      for idx in range(M - 1, -1, -1):
         if idx == M - 1:
            new_goals = vmap(transitions[idx], (0, 0))(pos, old_goals)
         else:
            cond = vmap(np.array_equal, (0, None))(old_goals, goals[idx])
            condition = np.stack((cond, cond), axis=1)
            new_goals = np.where(condition, vmap(transitions[idx], (0, 0))(pos, old_goals), new_goals)

      return new_goals

   return dgoal
