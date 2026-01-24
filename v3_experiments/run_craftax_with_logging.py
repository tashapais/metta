"""
Run Craftax experiments with full metric logging for generating plots.
Saves per-update metrics to numpy files for later visualization.
"""

import argparse
import time
import os
from functools import partial
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
from craftax.craftax_env import make_craftax_env_from_name


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    embedding: jnp.ndarray


class ActorCritic(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1)) if len(x.shape) > 2 else x.reshape((-1,)) if len(x.shape) == 1 else x
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        embedding = nn.Dense(self.embedding_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return logits, jnp.squeeze(value, axis=-1), embedding


class ProjectionHead(nn.Module):
    hidden_dim: int = 128
    output_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0))(x)
        return x


class GeometricContrastiveLoss:
    def __init__(self, temperature: float = 0.1, geometric_p: float = 0.1):
        self.temperature = temperature
        self.geometric_p = geometric_p

    def __call__(self, embeddings: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        batch_size, time_steps, embed_dim = embeddings.shape
        rng, offset_rng = jax.random.split(rng)
        offsets = jax.random.geometric(offset_rng, self.geometric_p, shape=(batch_size,))
        offsets = jnp.clip(offsets, 1, time_steps - 1)
        anchor_indices = jnp.zeros(batch_size, dtype=jnp.int32)
        positive_indices = jnp.minimum(offsets, time_steps - 1).astype(jnp.int32)
        anchors = embeddings[jnp.arange(batch_size), anchor_indices]
        positives = embeddings[jnp.arange(batch_size), positive_indices]
        anchors = anchors / (jnp.linalg.norm(anchors, axis=-1, keepdims=True) + 1e-8)
        positives = positives / (jnp.linalg.norm(positives, axis=-1, keepdims=True) + 1e-8)
        all_embeds = embeddings.reshape(-1, embed_dim)
        all_embeds = all_embeds / (jnp.linalg.norm(all_embeds, axis=-1, keepdims=True) + 1e-8)
        pos_sim = jnp.sum(anchors * positives, axis=-1) / self.temperature
        neg_sim = jnp.matmul(anchors, all_embeds.T) / self.temperature
        logsumexp_neg = jax.scipy.special.logsumexp(neg_sim, axis=-1)
        loss = -pos_sim + logsumexp_neg
        return jnp.mean(loss)


def make_train(config):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["num_updates"]
        return config["lr"] * frac

    def train(rng):
        rng, init_rng = jax.random.split(rng)
        network = ActorCritic(
            action_dim=env.action_space(env_params).n,
            hidden_dim=config["hidden_dim"],
            embedding_dim=config["embedding_dim"],
        )
        dummy_obs = jnp.zeros((1, env.observation_space(env_params).shape[0]))
        network_params = network.init(init_rng, dummy_obs)

        proj_head = ProjectionHead(hidden_dim=128, output_dim=config["embedding_dim"])
        rng, proj_rng = jax.random.split(rng)
        proj_params = proj_head.init(proj_rng, jnp.zeros((1, config["embedding_dim"])))

        all_params = {"network": network_params, "proj_head": proj_params}

        if config["anneal_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(config["lr"], eps=1e-5),
            )

        train_state = TrainState.create(apply_fn=network.apply, params=all_params, tx=tx)

        rng, env_rng = jax.random.split(rng)
        env_rng = jax.random.split(env_rng, config["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(env_rng, env_params)

        contrastive_loss_fn = GeometricContrastiveLoss(
            temperature=config["contrastive_temperature"],
            geometric_p=1 - config["contrastive_gamma"],
        )

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state
            rng, action_rng = jax.random.split(rng)
            logits, value, embedding = network.apply(train_state.params["network"], last_obs)
            action = jax.random.categorical(action_rng, logits)
            log_prob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
            rng, step_rng = jax.random.split(rng)
            step_rng = jax.random.split(step_rng, config["num_envs"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_rng, env_state, action, env_params
            )
            transition = Transition(
                done=done, action=action, value=value, reward=reward,
                log_prob=log_prob, obs=last_obs, next_obs=obsv, embedding=embedding,
            )
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])
            train_state, env_state, last_obs, rng = runner_state
            _, last_val, _ = network.apply(train_state.params["network"], last_obs)

            def _compute_gae(carry, transition):
                gae, next_value = carry
                delta = transition.reward + config["gamma"] * next_value * (1 - transition.done) - transition.value
                gae = delta + config["gamma"] * config["gae_lambda"] * (1 - transition.done) * gae
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(_compute_gae, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True)
            returns = advantages + traj_batch.value

            def _update_epoch(carry, unused):
                train_state, rng = carry
                rng, perm_rng = jax.random.split(rng)
                batch_size = config["num_envs"] * config["num_steps"]
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = (traj_batch, advantages, returns)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["num_minibatches"], -1) + x.shape[1:]), shuffled_batch
                )

                def _update_minibatch(train_state, minibatch):
                    traj, mb_advantages, mb_returns = minibatch

                    def _loss_fn(params):
                        logits, value, embedding = network.apply(params["network"], traj.obs)
                        log_prob = jax.nn.log_softmax(logits)[jnp.arange(traj.action.shape[0]), traj.action]
                        value_loss = 0.5 * jnp.mean((value - mb_returns) ** 2)
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        mb_advantages_normalized = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        pg_loss1 = -mb_advantages_normalized * ratio
                        pg_loss2 = -mb_advantages_normalized * jnp.clip(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"])
                        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))
                        entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1))

                        contrastive_loss = 0.0
                        if config["contrastive_coef"] > 0:
                            embed_for_contrast = embedding.reshape(config["num_envs"], -1, config["embedding_dim"])
                            projected = proj_head.apply(params["proj_head"], embed_for_contrast)
                            rng_contrast = jax.random.PRNGKey(0)
                            contrastive_loss = contrastive_loss_fn(projected, rng_contrast)

                        total_loss = pg_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
                        total_loss += config["contrastive_coef"] * contrastive_loss
                        return total_loss, (pg_loss, value_loss, entropy, contrastive_loss)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, aux), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, aux)

                train_state, losses = jax.lax.scan(_update_minibatch, train_state, minibatches)
                return (train_state, rng), losses

            (train_state, rng), losses = jax.lax.scan(_update_epoch, (train_state, rng), None, config["update_epochs"])

            metrics = {
                "total_loss": losses[0].mean(),
                "pg_loss": losses[1][0].mean(),
                "value_loss": losses[1][1].mean(),
                "entropy": losses[1][2].mean(),
                "contrastive_loss": losses[1][3].mean(),
                "mean_reward": traj_batch.reward.mean(),
            }
            return (train_state, env_state, last_obs, rng), metrics

        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["num_updates"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def run_experiment(config, seed=0):
    rng = jax.random.PRNGKey(seed)
    print(f"Creating training function...", flush=True)
    train_fn = make_train(config)
    print(f"JIT compiling...", flush=True)
    train_jit = jax.jit(train_fn)
    print(f"Starting training...", flush=True)
    start_time = time.time()
    out = train_jit(rng)
    print("Waiting for computation to complete...", flush=True)
    jax.block_until_ready(out)
    elapsed = time.time() - start_time

    # Extract all metrics as numpy arrays
    metrics_np = jax.tree_util.tree_map(lambda x: np.array(x), out["metrics"])

    print(f"Training completed in {elapsed:.2f}s", flush=True)
    print(f"Final reward: {metrics_np['mean_reward'][-1]:.4f}", flush=True)

    return metrics_np, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--contrastive_coef", type=float, default=0.001)
    parser.add_argument("--output_dir", type=str, default="v3_experiments/metrics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        "lr": 2.5e-4,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "num_minibatches": 4,
        "update_epochs": 4,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "anneal_lr": True,
        "hidden_dim": 256,
        "embedding_dim": 64,
        "contrastive_coef": args.contrastive_coef if args.contrastive else 0.0,
        "contrastive_temperature": 0.1,
        "contrastive_gamma": 0.977,
    }

    batch_size = config["num_envs"] * config["num_steps"]
    config["num_updates"] = args.total_timesteps // batch_size

    method_name = "contrastive" if args.contrastive else "ppo"
    print("=" * 60, flush=True)
    print(f"Running: {method_name.upper()}", flush=True)
    print("=" * 60, flush=True)

    all_metrics = []
    for seed in range(args.num_seeds):
        print(f"\n--- Seed {seed} ---", flush=True)
        metrics, elapsed = run_experiment(config, seed=seed)
        all_metrics.append(metrics)

        # Save per-seed metrics
        np.savez(
            f"{args.output_dir}/{method_name}_seed{seed}.npz",
            **metrics
        )

    # Stack all seeds
    stacked = {k: np.stack([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    np.savez(f"{args.output_dir}/{method_name}_all_seeds.npz", **stacked)

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    final_rewards = [m["mean_reward"][-1] for m in all_metrics]
    print(f"Final reward: {np.mean(final_rewards):.4f} +/- {np.std(final_rewards):.4f}", flush=True)


if __name__ == "__main__":
    main()
