import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import optax
import argparse

from config import Config
from functions import ForceNet
from functions import TrueForceNet
from dataloader import data_loader_tp_next, data_loader_traj_next
from serialization import save

jax.config.update("jax_debug_nans", True)



# loss fn for whole trajectory
def single_traj_generator(model, pedestrian_idx, position, other_positions, velocity, other_velocities, dt):
    ped_force = jax.vmap(model.pedestrian_force, in_axes=(0, 0))
    other_trajs = jnp.stack((other_positions, other_velocities), axis=1)

    def propagator(curr_data, other_data):
        curr_position, curr_velocity, pedestrian_idx, dt = curr_data

        rel_disp = curr_position - other_data[0]
        rel_vel = curr_velocity - other_data[1]

        f = ped_force(rel_disp, rel_vel)
        goal_f = model.goal_force(pedestrian_idx, curr_velocity)
        new_velocity = (goal_f + jnp.sum(f, axis=0)) * dt + curr_velocity
        new_position = curr_position + (new_velocity + curr_velocity) * dt / 2
        return (new_position, new_velocity, pedestrian_idx, dt), jnp.array([new_position, new_velocity])

    _, predicted_traj = jax.lax.scan(propagator, (position, velocity, pedestrian_idx, dt), other_trajs)

    return jnp.concat((jnp.array([[position, velocity]]), predicted_traj[:-1]))

def single_tp_ldad_loss_fn(state, pred_state):
    k_1 = 0.5
    k_2 = 0.5

    norm_state = jnp.linalg.norm(state)
    norm_pred_state = jnp.linalg.norm(pred_state)
    LD = jnp.sqrt((norm_state - norm_pred_state) ** 2) / jnp.where(norm_state + norm_pred_state < 0.001, 0.001, norm_state + norm_pred_state)
    AD = (1 - jnp.dot(state, pred_state) / jnp.where(norm_state * norm_pred_state < 0.001, 0.001, norm_state * norm_pred_state)) / 2
    return k_1 * LD + k_2 * AD

def single_traj_loss_fn(model, pedestrian_idx, positions, other_positions, velocities, other_velocities, dt):
    traj = jnp.stack((positions, velocities), axis=1)
    pred_traj = single_traj_generator(model, pedestrian_idx, positions[0], other_positions, velocities[0], other_velocities, dt)

    # return jnp.sum(jax.vmap(single_tp_ldad_loss_fn)(positions, pred_traj[:, 0]))
    # return jnp.sum(jax.vmap(single_tp_ldad_loss_fn)(traj, pred_traj))
    # return jnp.linalg.norm(pred_traj[:, 1] - velocity) ** 2
    return jnp.linalg.norm(pred_traj[:, 0] - positions) ** 2


def batch_traj_loss_fn(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, dt):
    loss_fn = jax.vmap(single_traj_loss_fn, in_axes=(None, 0, 0, 0, 0, 0, None))
    return jnp.sum(loss_fn(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, dt))

@eqx.filter_jit
def traj_make_step(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, dt, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_traj_loss_fn)
    loss, grads = loss_fn(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, dt)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

@eqx.filter_jit
def traj_eval_step(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, dt):
    loss = batch_traj_loss_fn(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, dt)
    return loss


# fix: timeframe,
def main(cfg: Config):

    # to mimic tp loss
    # timeframe = 2

    timeframe = int(1.5/cfg.dt)

    key = jr.PRNGKey(cfg.seed)
    model_key, train_key = jr.split(key)
    if cfg.init_goal_vel_path is not None:
        goal_velocities = jnp.load(cfg.init_goal_vel_path)
    else:
        goal_velocities = jnp.zeros((cfg.num_pedestrians, 2))

    # model = ForceNet(model_key, goal_velocities, cfg.pedestrian_hidden_sizes, cfg.goal_hidden_sizes)
    model = TrueForceNet(goal_velocities, tau=jnp.array(0.0), A=jnp.array(0.0), d0=jnp.array(0.0), B=jnp.array(0.0))
    opt = optax.adam(learning_rate=cfg.learning_rate, b1=cfg.beta1, b2=cfg.beta2)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    training_losses = []
    evaluation_losses = []

    dataset = jnp.load(cfg.dataset_path)

    assert cfg.dt == dataset["dt"]

    positions = dataset["positions"][100:]
    velocities = dataset["velocities"][100:]
    # Split into train and eval sets along the first dimension (time)
    num_timesteps = positions.shape[0]
    num_eval = int(num_timesteps * cfg.eval_fraction)
    num_train = num_timesteps - num_eval

    # Don't shuffle time indices
    train_idx = jnp.arange(num_train)
    eval_idx = jnp.arange(num_train, num_timesteps)
    # eval_idx = jnp.arange(num_eval)
    # train_idx = jnp.arange(num_eval, num_timesteps)

    # Apply split
    train_positions = positions[train_idx]
    train_velocities = velocities[train_idx]
    eval_positions = positions[eval_idx]
    eval_velocities = velocities[eval_idx]

    for i in range(cfg.num_epochs):

        train_key, val_key = jr.split(train_key)
        train_loader = data_loader_traj_next(train_positions, train_velocities,
                                        batch_size=cfg.batch_size,
                                        timeframe=timeframe,
                                        rng_key=train_key,
                                        shuffle=True, drop_last=True)

        # Training loop: update model but don't collect losses yet
        for batch in train_loader:
            _, model, opt_state = traj_make_step(
                model,
                batch["person_index"],
                batch["pos"],
                batch["others_pos"],
                batch["vel"],
                batch["others_vel"],
                cfg.dt,
                opt_state, opt.update)

        # Compute training loss on final model state (like evaluation)
        if i % cfg.log_interval == 0:
            train_loader_eval = data_loader_traj_next(train_positions, train_velocities,
                                                    batch_size=cfg.batch_size, timeframe=timeframe,
                                                    rng_key=train_key, shuffle=False, drop_last=False)

            train_losses = []
            for batch in train_loader_eval:
                train_loss = traj_eval_step(model, batch["person_index"], batch["pos"], batch["others_pos"], batch["vel"], batch["others_vel"], cfg.dt)
                train_losses.append(train_loss)

            avg_train_loss = jnp.mean(jnp.stack(train_losses))
            training_losses.append(avg_train_loss)
            print(f"Epoch {i}, Training Loss: {avg_train_loss}")

        if i % cfg.eval_interval == 0:
            eval_loader = data_loader_traj_next(eval_positions, eval_velocities,
                                      batch_size=cfg.batch_size, timeframe=timeframe,
                                      rng_key=val_key, shuffle=False, drop_last=False)

            eval_losses = []
            true_trajs = []
            pred_trajs = []
            for batch in eval_loader:
                eval_loss = traj_eval_step(model, batch["person_index"], batch["pos"], batch["others_pos"], batch["vel"], batch["others_vel"], cfg.dt)
                eval_losses.append(eval_loss)

                # debug code attempt
                true_traj = batch["pos"][0]
                pred_traj = single_traj_generator(model, batch["person_index"][0], batch["pos"][0, 0], batch["others_pos"][0], batch["vel"][0, 0], batch["others_vel"][0], cfg.dt)[:, 0]
                assert true_traj.shape == pred_traj.shape, f"Different trajectory shapes: Got {true_traj.shape} and {pred_traj.shape}."
                true_trajs.append(true_traj)
                pred_trajs.append(pred_traj)

            avg_eval_loss = jnp.mean(jnp.stack(eval_losses))
            evaluation_losses.append(avg_eval_loss)
            print(f"Epoch {i}, Evaluation Loss: {avg_eval_loss}")

            # debug code attempt
            fig, ax = plt.subplots()
            for traj in true_trajs:
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.1, color="blue")
            for traj in pred_trajs:
                ax.plot(traj[:, 0], traj[:, 1], alpha=1, color="blue")
            plt.savefig(f"true_vs_pred_trajs_{i}.png")
            plt.close(fig)




    jnp.save(f"training_losses_{cfg.experiment_name}.npy", training_losses)
    jnp.save(f"evaluation_losses_{cfg.experiment_name}.npy", evaluation_losses)
    save(f"model_traj_{cfg.experiment_name}.eqx", cfg, model)

# ----------------------------- Single step prediction -----------------------------

# this is loss function for a single step, also write a loss function for a whole trajectory
def single_loss_fn(model, pedestrian_idx, position, other_positions, velocity, other_velocities, y_velocity, dt):
    rel_disp = position - other_positions
    rel_vel = velocity - other_velocities
    f = jax.vmap(model.pedestrian_force, in_axes=(0, 0))(rel_disp, rel_vel)
    goal_f = model.goal_force(pedestrian_idx, velocity)
    return jnp.linalg.norm((goal_f + jnp.sum(f, axis=0))*dt + velocity - y_velocity)**2

def batch_loss_fn(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, y_velocities, dt):
    loss_fn = jax.vmap(single_loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, None))
    return jnp.sum(loss_fn(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, y_velocities, dt))
    # return jnp.mean(loss_fn(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, y_velocities, dt))

@eqx.filter_jit
def make_step(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, y_velocities, dt, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, y_velocities, dt)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

@eqx.filter_jit
def eval_step(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, y_velocities, dt):
    loss = batch_loss_fn(model, pedestrian_indices, positions, other_positions, velocities, other_velocities, y_velocities, dt)
    return loss

def _main(cfg: Config):

    key = jr.PRNGKey(cfg.seed)
    model_key, train_key = jr.split(key)
    if cfg.init_goal_vel_path is not None:
        goal_velocities = jnp.load(cfg.init_goal_vel_path)
    else:
        goal_velocities = jnp.zeros((cfg.num_pedestrians, 2))

    model = ForceNet(model_key, goal_velocities, cfg.pedestrian_hidden_sizes, cfg.goal_hidden_sizes)
    # model = TrueForceNet(goal_velocities, tau=jnp.array(0.0), A=jnp.array(0.0), d0=jnp.array(0.0), B=jnp.array(0.0))
    opt = optax.adam(learning_rate=cfg.learning_rate, b1=cfg.beta1, b2=cfg.beta2)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    training_losses = []
    evaluation_losses = []

    dataset = jnp.load(cfg.dataset_path)

    assert cfg.dt == dataset["dt"]

    positions = dataset["positions"][100:]
    velocities = dataset["velocities"][100:]
    # Split into train and eval sets along the first dimension (time)
    num_timesteps = positions.shape[0]
    num_eval = int(num_timesteps * cfg.eval_fraction)
    num_train = num_timesteps - num_eval

    # Don't shuffle time indices
    train_idx = jnp.arange(num_train)
    eval_idx = jnp.arange(num_train, num_timesteps)
    # eval_idx = jnp.arange(num_eval)
    # train_idx = jnp.arange(num_eval, num_timesteps)

    # Apply split
    train_positions = positions[train_idx]
    train_velocities = velocities[train_idx]
    eval_positions = positions[eval_idx]
    eval_velocities = velocities[eval_idx]

    for i in range(cfg.num_epochs):

        train_key, val_key = jr.split(train_key)
        train_loader = data_loader_tp_next(train_positions, train_velocities,
                                        batch_size=cfg.batch_size,
                                        rng_key=train_key,
                                        shuffle=True, drop_last=True)

        # Training loop: update model but don't collect losses yet
        for batch in train_loader:
            _, model, opt_state = make_step(
                model,
                batch["person_index"],
                batch["pos"],
                batch["others_pos"],
                batch["vel"],
                batch["others_vel"],
                batch["next_vel"],
                cfg.dt,
                opt_state, opt.update)

        # Compute training loss on final model state (like evaluation)
        if i % cfg.log_interval == 0:
            train_loader_eval = data_loader_tp_next(train_positions, train_velocities,
                                                    batch_size=cfg.batch_size,
                                                    rng_key=train_key, shuffle=False, drop_last=False)

            train_losses = []
            for batch in train_loader_eval:
                train_loss = eval_step(model, batch["person_index"], batch["pos"], batch["others_pos"], batch["vel"], batch["others_vel"], batch["next_vel"], cfg.dt)
                train_losses.append(train_loss)

            avg_train_loss = jnp.mean(jnp.stack(train_losses))
            training_losses.append(avg_train_loss)
            print(f"Epoch {i}, Training Loss: {avg_train_loss}")

        if i % cfg.eval_interval == 0:
            eval_loader = data_loader_tp_next(eval_positions, eval_velocities,
                                      batch_size=cfg.batch_size,
                                      rng_key=val_key, shuffle=False, drop_last=False)

            eval_losses = []
            for batch in eval_loader:
                eval_loss = eval_step(model, batch["person_index"], batch["pos"], batch["others_pos"], batch["vel"], batch["others_vel"], batch["next_vel"], cfg.dt)
                eval_losses.append(eval_loss)

            avg_eval_loss = jnp.mean(jnp.stack(eval_losses))
            evaluation_losses.append(avg_eval_loss)
            print(f"Epoch {i}, Evaluation Loss: {avg_eval_loss}")

    jnp.save(f"training_losses_{cfg.experiment_name}.npy", training_losses)
    jnp.save(f"evaluation_losses_{cfg.experiment_name}.npy", evaluation_losses)
    save(f"model_{cfg.experiment_name}.eqx", cfg, model)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train pedestrian dynamics model")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    # parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--num_pedestrians", type=int, default=200)
    # parser.add_argument("--eval_fraction", type=float, default=0.02)
    parser.add_argument("--eval_fraction", type=float, default=0.025)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--dataset_path", type=str, default="pedestrians.npz")
    parser.add_argument("--init_goal_vel_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)


    args = parser.parse_args()

    cfg = Config(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        num_pedestrians=args.num_pedestrians,
        eval_fraction=args.eval_fraction,
        dt=args.dt,
        experiment_name=args.experiment_name,
        dataset_path=args.dataset_path,
        init_goal_vel_path=args.init_goal_vel_path,
        seed=args.seed,
    )

    print("Using config:")
    print(cfg)

    main(cfg)
