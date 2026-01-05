import json
import dataclasses
import os
import re

import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax

from config import Config
from functions import ForceNet
from functions import TrueForceNet

# modified from https://docs.kidger.site/equinox/examples/serialisation/


def make(key: jax.Array, cfg: Config):
    if cfg.init_goal_vel_path is not None:
        goal_velocities = jnp.load(cfg.init_goal_vel_path)
    else:
        goal_velocities = jnp.zeros((cfg.num_pedestrians, 2))
    # model = ForceNet(key, goal_velocities, cfg.pedestrian_hidden_sizes, cfg.goal_hidden_sizes)
    model = TrueForceNet(goal_velocities, tau=0.0, A=0.0, d0=0.0, B=0.0)
    return model


def save(filename, config, model):
    with open(filename, 'wb') as f:
        hyperparam_str = json.dumps(dataclasses.asdict(config))
        f.write((hyperparam_str + '\n').encode())
        eqx.tree_serialise_leaves(f, model)

def load(filename):

    with open(filename, 'rb') as f:
        hyperparams = json.loads(f.readline().decode())
        cfg = Config(**hyperparams)

        model = make(jr.PRNGKey(0), cfg=cfg)

        model_new = eqx.tree_deserialise_leaves(f, model)
        return model_new
