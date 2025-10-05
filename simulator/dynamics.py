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
from .utils import normal, angle_correct, init_goal_speed
from .force import align_tot


# CHIRAL PARTICLES


class ChiralAMState(namedtuple('ChiralAMState', ['position', 'theta', 'speed', 'omega', 'key'])):
    pass

def chiral_am(shift_fn, displacement_fn, dt, N, box_size, g = 0.018, D_r = 0.009):
    """
    Simulation of chiral active particle dynamics.

    The default values of the simulation function are provided in [1].

    Inputs:
        shift_fn (func): A function that displaces positions, `R`, by an amount `dR`.
            Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
        displacement_fn (func):  A function that computes displacements between R1 and R2.
            Both R1, R2 should be ndarrays of shape `[n, spatial_dimension]`
        dt (float): Floating point number specifying the timescale (step size) of the
            simulation.
        N (int): Integer number specifying the number of particles in the simulation
        box_size (float): Floating point number specifying the length of the
            square box simulation.
        g (float): Floating point number specifying the strength of the polar alignment
            interactions
        D_r (float): Floating point number specifying the rotational diffusion constant

    Outputs:
        init_fn, step_fn (funcs): functions to initialize the simulation and to timestep
            the simulation

    [1] R. Supekar, B. Song, A. Hastewell. "Learning hydrodynamic equations for active matter from particle
        simulations and experiments".
    """
    def init_fn(key, mu_v = 1.0, sigma_v = 0.4, mu_r = 2.2, sigma_r = 1.7, cutoff_omega = 1.4):
        """
        Initializes speed v and rotational radius r in a Gaussian distribution.
        Angular velocity omega is determined by the relation v = omega * r
        """
        key, v_key, omega_key, R_key, theta_key = random.split(key, 5)

        # SPEED AND ANGULAR VELOCITY INIT
        # sample from gaussian distrib + filter for positive vals
        spd = mu_v + sigma_v * random.normal(v_key, (10 * N, ))
        r = mu_r + sigma_r * random.normal(omega_key, (10 * N, ))

        condition1 = np.where(spd > 0)[0]
        spd = spd[condition1]
        r = r[condition1]

        condition2 = np.where(r > 0)[0]
        spd = spd[condition2]
        r = r[condition2]
        omega = np.divide(spd, r)

        condition3 = np.where(omega <= cutoff_omega)[0]
        spd = spd[condition3]
        omega = omega[condition3]

        spd = spd[:N]
        omega = omega[:N]

        # COORDS INIT
        R = box_size * random.uniform(R_key, (N, 2))
        theta = random.uniform(theta_key, (N,), maxval=2*onp.pi)

        state = ChiralAMState(R, theta, spd, omega, key)

        return state

    @jit
    def step_fn(_, state):
        # Forward-Euler scheme
        n = normal(state.speed, state.theta)
        dR = n * dt
        dtheta = (state.omega + g * align_tot(state.position, state.theta, displacement_fn)) * dt

        # Stochastic step:
        key, split = random.split(state.key)
        stheta = np.sqrt(2 * D_r * dt) * random.normal(split, state.theta.shape)

        return ChiralAMState(shift_fn(state.position, dR),
                             angle_correct(state.theta + dtheta + stheta),
                             state.speed,
                             state.omega,
                             key)

    return init_fn, step_fn


# PEDESTRIAN DYNAMICS


class PedestrianState(namedtuple("PedestrianState", ['position', 'velocity', 'radius', 'goal_speed', 'goal_orientation', 'key'])):
    def speed(self):
        """Returns the speed of all pedestrians"""
        return np.sqrt(self.velocity[:, 0] ** 2 + self.velocity[:, 1] ** 2)

    def orientation(self):
        """Returns the orientation angle of all pedestrians"""
        return angle_correct(np.atan2(self.velocity[:, 1], self.velocity[:, 0]))

    def kinetic(self):
        """Returns the kinetic energy of all pedestrians"""
        return (np.linalg.norm(self.velocity, axis=1) ** 2) / 2

def orient_goal(goal, position):
    # updates new goal orientation to point towards goal point:
    disp_to_goal = goal - position
    dist_to_goal = np.linalg.norm(disp_to_goal, axis=1)
    _dist_to_goal = np.stack([dist_to_goal, dist_to_goal], axis=1)
    unit_disp_to_goal = disp_to_goal / _dist_to_goal
    goal_orient = np.arctan2(unit_disp_to_goal[:, 1], unit_disp_to_goal[:, 0])

    return goal_orient


class ExperimentalPedestrianState(namedtuple("PedestrianState", ['position', 'velocity', 'radius', 'goal_speed', 'key', 'goal', 'goal_orientation'])):
    """Implements state of pedestrians, using goal points instead of goal directions

    Args:
        position: (N, 2) array of positions
        velocity: (N, 2) array of velocities
        radius: 1 scalar of particle radius
        goal_speed: (1,)/(N, 1) array of goal speeds
        key: RNG key for stochasticity

    Optional args:
        goal: (N, 2) list of goals for each particle
    The functions decides when a target is reached, and decides the new target then.
    """
    def speed(self):
        """Returns the speed of all pedestrians"""
        return np.sqrt(self.velocity[:, 0] ** 2 + self.velocity[:, 1] ** 2)

    def orientation(self):
        """Returns the orientation angle of all pedestrians"""
        return angle_correct(np.atan2(self.velocity[:, 1], self.velocity[:, 0]))

    def kinetic(self):
        """Returns the kinetic energy of all pedestrians"""
        return (np.linalg.norm(self.velocity, axis=1) ** 2) / 2


class Wall(object):
    pass

class StraightWall(Wall):
    def __init__(self, start, end):
        self.start = start
        self.end = end

class CircleWall(Wall):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

def pedestrian(shift_fn, force_fn, dt, N, **sim_kwargs):
    """
    Simulation of pedestrian models

    Inputs:
        shift_fn (func)             : returned by jax_md.space
        displacement_fn (func)      : returned by jax_md.space
        force_fn (func) : a function characterizing the interaction between
                                    pedestrians
        dt (float)      : Floating point number specifying the timescale (step size) of the simulation.
        N (int)         : Integer number specifying the number of particles in the simulation

    Extra inputs:
        stochastic (bool)   : True if simulation should be stochastic. Defaults to True
        experimental (bool) : True to toggle some experimental features. Defaults to False

    Outputs:
        init_fn, step_fn (funcs): functions to initialize the simulation and to timestep
            the simulation
    """
    # force_fn = jit(quantity.canonicalize_force(energy_or_force_fn))
    # force_fn = energy_or_force_fn

    if 'stochastic' in sim_kwargs:
        stochastic = sim_kwargs['stochastic']
    else:
        stochastic = True

    if 'experimental' in sim_kwargs:
        experimental = sim_kwargs['experimental']
    else:
        experimental = False

    if not experimental:
        def init_fn(pos, radius, **kwargs):
            """
            Initializes a pedestrian simulation.

            Inputs:
                pos (Array): position of all pedestrians
                radius (float): collision radius of pedestrians

            Extra Inputs:
                key (RNG key)
                goal_speed (Array)
                goal_orientation (Array)
                velocity (Array)

            Output:
                A PedestrianState instance.
            """
            key = kwargs['key'] if 'key' in kwargs else None

            if 'goal_speed' in kwargs:
                goal_speed = kwargs['goal_speed']
            else:
                if key is None:
                    raise ValueError("PRNG key required for initializing goal speed")
                key, split = random.split(key)
                goal_speed = init_goal_speed(split, N)

            if 'goal_orientation' in kwargs:
                goal_orientation = kwargs['goal_orientation']
            else:
                goal_orientation = None

            if 'velocity' in kwargs:
                velocity = kwargs['velocity']
            elif goal_orientation is not None:
                velocity = normal(goal_speed, goal_orientation)
            else:
                if key is None:
                    raise ValueError("PRNG key required for initializing goal speed")
                key, split = random.split(key)
                orient = random.uniform(split, (N,), minval=0, maxval=2 * onp.pi)
                velocity = normal(goal_speed, orient)

            return PedestrianState(pos, velocity, radius, goal_speed, goal_orientation, key)

        def step_fn(_, state):
            dstate = force_fn(state)

            # stochastic impl
            if stochastic:
                key, split = random.split(state.key)
                svelocity = random.uniform(split, (N, 2), minval=-1.0, maxval=1.0)
            else:
                key = state.key
                svelocity = np.zeros((N, 2))

            rad = dstate.radius if dstate.radius is not None else state.radius
            goal_speed = dstate.goal_speed if dstate.goal_speed is not None else state.goal_speed
            goal_orientation = dstate.goal_orientation if dstate.goal_orientation is not None else state.goal_orientation

            return PedestrianState(
                shift_fn(state.position, state.velocity * dt),
                state.velocity + (dstate.position + svelocity) * dt,
                rad,
                goal_speed,
                goal_orientation,
                key
            )

    else:
        def init_fn(pos, radius, **kwargs):
            """
            Initializes a pedestrian simulation.

            Inputs:
                pos (Array): position of all pedestrians
                radius (float): collision radius of pedestrians

            Extra Inputs:
                key (RNG key)
                goal_speed (Array)
                velocity (Array)
                goal (list(list(list(float, float))))
                status (list(int))
                condition (list(functions(idx, state)))

            Output:
                An ExperimentalPedestrianState instance.
            """
            key = kwargs['key'] if 'key' in kwargs else None

            if 'goal_speed' in kwargs:
                goal_speed = kwargs['goal_speed']
            else:
                if key is None:
                    raise ValueError("PRNG key required for initializing goal speed")
                key, split = random.split(key)
                goal_speed = init_goal_speed(split, N)

            if 'goal' in kwargs:
                goal = kwargs['goal']
            else:
                goal = None

            # if 'condition' in kwargs:
            #     condition = kwargs['condition']
            # else:
            #     condition = None if goal is None else [lambda idx, state: idx for _ in goal]

            if goal is None:
                if 'goal_orientation' in kwargs:
                    goal_orientation = kwargs['goal_orientation']
                else:
                    goal_orientation = None
            else:
                goal_orientation = orient_goal(goal, pos)

            if 'velocity' in kwargs:
                velocity = kwargs['velocity']
            elif goal_orientation is not None:
                velocity = normal(goal_speed, goal_orientation)
            else:
                if key is None:
                    raise ValueError("PRNG key required for initializing goal speed")
                key, split = random.split(key)
                orient = random.uniform(split, (N,), minval=0, maxval=2 * onp.pi)
                velocity = normal(goal_speed, orient)

            return ExperimentalPedestrianState(pos, velocity, radius, goal_speed, key, goal, goal_orientation)

        def step_fn(_, state):
            dstate = force_fn(state)

            # stochastic impl
            if stochastic:
                key, split = random.split(state.key)
                svelocity = random.uniform(split, (N, 2), minval=-1.0, maxval=1.0)
            else:
                key = state.key
                svelocity = np.zeros((N, 2))

            new_pos = shift_fn(state.position, state.velocity * dt)
            new_velocity = state.velocity + (dstate.position + svelocity) * dt
            new_goal = dstate.goal if dstate.goal is not None else state.goal

            if new_goal is not None:
                new_goal_orientation = orient_goal(new_goal, new_pos)
            else:
                new_goal_orientation = state.goal_orientation

            return ExperimentalPedestrianState(
                new_pos,
                new_velocity,
                state.radius,
                state.goal_speed,
                key,
                new_goal,
                new_goal_orientation
            )

    return init_fn, step_fn
