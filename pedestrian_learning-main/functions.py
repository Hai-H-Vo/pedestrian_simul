import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

class PedestrianForceNet(eqx.Module):
    layers: list
    final_layer: eqx.nn.Linear

    def __init__(self, key: jax.Array, hidden_sizes=(64, 64)):
        
        keys = jr.split(key, len(hidden_sizes) + 1)
        layers = []
        in_size = 4  # [disp_x, disp_y, rel_vx, rel_vy]
        for i, h in enumerate(hidden_sizes):
            layers.append(eqx.nn.Linear(in_size, h, key=keys[i]))
            layers.append(jax.nn.tanh)
            in_size = h
        self.layers = layers
        self.final_layer = eqx.nn.Linear(in_size, 2, key=keys[-1])

    def __call__(self, rel_disp: jax.Array, rel_vel: jax.Array) -> jax.Array:
        x = jnp.concatenate([rel_disp, rel_vel], axis=-1)
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

class GoalForceNet(eqx.Module):
    layers: list
    final_layer: eqx.nn.Linear

    def __init__(self, key: jax.Array, hidden_sizes=(64, 64)):
        keys = jr.split(key, len(hidden_sizes) + 1)
        layers = []
        in_size = 4  # [vx, vy, vx_goal, vy_goal]
        for i, h in enumerate(hidden_sizes):
            layers.append(eqx.nn.Linear(in_size, h, key=keys[i]))
            layers.append(jax.nn.tanh)
            in_size = h
        self.layers = layers
        self.final_layer = eqx.nn.Linear(in_size, 2, key=keys[-1])

    def __call__(self, velocity: jax.Array, goal_velocity: jax.Array) -> jax.Array:
        x = jnp.concatenate([velocity, goal_velocity], axis=-1)
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

class ForceNet(eqx.Module):
    pedestrian_force_net: PedestrianForceNet
    goal_force_net: GoalForceNet
    goal_velocities: jax.Array # = eqx.field(static=True)

    def __init__(self, key: jax.Array, goal_velocities: jax.Array, pedestrian_hidden_sizes: list[int], goal_hidden_sizes: list[int]):
        pedestrian_key, goal_key = jr.split(key, 2)
        self.pedestrian_force_net = PedestrianForceNet(pedestrian_key, pedestrian_hidden_sizes)
        self.goal_force_net = GoalForceNet(goal_key, goal_hidden_sizes)
        self.goal_velocities = goal_velocities

    def __call__(self, pedestrian_idx: int, velocity: jax.Array, displacement: jax.Array, relative_velocity: jax.Array) -> jax.Array:
        pedestrian_f = self.pedestrian_force_net(displacement, relative_velocity)
        goal_f = self.goal_force_net(velocity, self.goal_velocities[pedestrian_idx])
        return pedestrian_f + goal_f

    def pedestrian_force(self, displacement: jax.Array, relative_velocity: jax.Array) -> jax.Array:
        return self.pedestrian_force_net(displacement, relative_velocity)

    def goal_force(self, pedestrian_idx: int, velocity: jax.Array) -> jax.Array:
        return self.goal_force_net(velocity, self.goal_velocities[pedestrian_idx])

class TrueForceNet(eqx.Module):
    goal_velocities: jax.Array
    tau: jax.Array
    A: jax.Array
    d0: jax.Array
    B: jax.Array
    def __init__(self, goal_velocities: jax.Array, tau: jax.Array, A: jax.Array, d0: jax.Array, B: jax.Array):
        self.A = A
        self.d0 = d0
        self.B = B
        self.tau = tau
        self.goal_velocities = goal_velocities

    def __call__(self, pedestrian_idx: int, velocity: jax.Array, displacement: jax.Array, relative_velocity: jax.Array) -> jax.Array:
        goal_f = self.goal_force(pedestrian_idx, velocity)
        pedestrian_f = self.pedestrian_force(displacement, relative_velocity)
        return goal_f + pedestrian_f

    def goal_force(self, pedestrian_idx: int, velocity: jax.Array) -> jax.Array:
        return (self.goal_velocities[pedestrian_idx] - velocity) / jnp.exp(self.tau)

    def pedestrian_force(self, displacement: jax.Array, relative_velocity: jax.Array) -> jax.Array:
        return jnp.exp(self.A) * jnp.exp((jnp.exp(self.d0) - jnp.linalg.norm(displacement)) / jnp.exp(self.B)**2) * (displacement / jnp.linalg.norm(displacement))