"""
Craftax experiments with geometric diagnostic metrics.

Paper experiments addressed:
  Exp 3: Horizon scaling (horizons 100, 500, 2000 × lambda 0.95, 0.99 × 5 seeds)
  Exp 4: Embedding trajectory visualization (Craftax part)
  Exp 5: SVD spectrum analysis (Craftax part)
  Exp 6: Cross-environment consistency (Craftax part)

Usage:
    # Single run
    CUDA_VISIBLE_DEVICES=1 python run_craftax_geom.py --num_steps 100 --gae_lambda 0.95 --seed 0

    # All Exp 3 runs
    CUDA_VISIBLE_DEVICES=1 python run_craftax_geom.py --run_all
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import numpy as np
from typing import NamedTuple
import time
import argparse
import json
import os

import wandb

from craftax.craftax_env import make_craftax_env_from_name
from geometric_metrics import compute_geometric_metrics_jax, effective_rank, expansion_ratio


class ActorCritic(nn.Module):
    """Actor-critic with exposed critic hidden layer."""
    action_dim: int
    hidden_dim: int = 256
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1)).astype(jnp.float32)

        # Shared encoder
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Embedding
        embedding = nn.Dense(self.embedding_dim)(x)

        # Actor
        actor = nn.Dense(self.hidden_dim)(x)
        actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim)(actor)

        # Critic with exposed hidden
        critic_h = nn.Dense(self.hidden_dim)(x)
        critic_h = nn.relu(critic_h)
        value = nn.Dense(1)(critic_h)

        return logits, value.squeeze(-1), embedding, critic_h


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    embedding: jnp.ndarray
    critic_hidden: jnp.ndarray


def make_train(config):
    """Create training function with geometric metric extraction."""

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    action_dim = env.action_space(env_params).n

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["num_updates"]
        return config["lr"] * frac

    def train(rng):
        rng, init_rng = jax.random.split(rng)
        network = ActorCritic(
            action_dim=action_dim,
            hidden_dim=config["hidden_dim"],
            embedding_dim=config["embedding_dim"],
        )

        rng, obs_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1,) + env.observation_space(env_params).shape)
        params = network.init(init_rng, dummy_obs)

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

        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, config["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state

            def _env_step(carry, unused):
                train_state, env_state, last_obs, rng = carry
                rng, action_rng, step_rng = jax.random.split(rng, 3)

                logits, value, embedding, critic_h = network.apply(train_state.params, last_obs)
                action_dist = jax.nn.softmax(logits)
                action = jax.random.categorical(action_rng, logits)
                log_prob = jnp.log(action_dist[jnp.arange(config["num_envs"]), action] + 1e-8)

                step_rng = jax.random.split(step_rng, config["num_envs"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    step_rng, env_state, action, env_params
                )

                transition = Transition(
                    done=done, action=action, value=value, reward=reward,
                    log_prob=log_prob, obs=last_obs, embedding=embedding,
                    critic_hidden=critic_h,
                )
                return (train_state, env_state, obsv, rng), transition

            runner_state_inner, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, config["num_steps"]
            )
            train_state, env_state, last_obs, rng = runner_state_inner

            _, last_val, _, _ = network.apply(train_state.params, last_obs)

            def _calculate_gae(carry, transition):
                gae, next_value = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["gamma"] * next_value * (1 - done) - value
                gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _calculate_gae,
                (jnp.zeros(config["num_envs"]), last_val),
                traj_batch,
                reverse=True,
            )
            returns = advantages + traj_batch.value

            def _update_epoch(carry, unused):
                train_state, rng = carry
                rng, perm_rng = jax.random.split(rng)

                batch_size = config["num_steps"] * config["num_envs"]
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                advantages_flat = advantages.reshape(batch_size)
                returns_flat = returns.reshape(batch_size)

                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = jax.tree_util.tree_map(lambda x: x[permutation], batch)
                advantages_flat = advantages_flat[permutation]
                returns_flat = returns_flat[permutation]

                def _update_minibatch(train_state, batch_info):
                    mb_batch, mb_advantages, mb_returns = batch_info

                    def loss_fn(params):
                        logits, value, embedding, critic_h = network.apply(params, mb_batch.obs)
                        action_dist = jax.nn.softmax(logits)
                        log_prob = jnp.log(action_dist[jnp.arange(len(mb_batch.action)), mb_batch.action] + 1e-8)
                        ratio = jnp.exp(log_prob - mb_batch.log_prob)
                        mb_adv_normalized = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        pg_loss1 = -mb_adv_normalized * ratio
                        pg_loss2 = -mb_adv_normalized * jnp.clip(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"])
                        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
                        value_loss = 0.5 * ((value - mb_returns) ** 2).mean()
                        entropy = -jnp.sum(action_dist * jnp.log(action_dist + 1e-8), axis=-1).mean()
                        total_loss = pg_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
                        return total_loss, (pg_loss, value_loss, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (loss, aux), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, aux)

                minibatch_size = batch_size // config["num_minibatches"]
                mb_idxs = jnp.arange(batch_size).reshape(config["num_minibatches"], minibatch_size)
                minibatches = jax.tree_util.tree_map(lambda x: x[mb_idxs], batch)
                mb_advantages = advantages_flat[mb_idxs]
                mb_returns = returns_flat[mb_idxs]

                train_state, losses = jax.lax.scan(
                    _update_minibatch, train_state, (minibatches, mb_advantages, mb_returns)
                )
                return (train_state, rng), losses

            (train_state, rng), losses = jax.lax.scan(
                _update_epoch, (train_state, rng), None, config["update_epochs"]
            )

            metrics = {
                "total_loss": losses[0].mean(),
                "pg_loss": losses[1][0].mean(),
                "value_loss": losses[1][1].mean(),
                "entropy": losses[1][2].mean(),
                "mean_reward": traj_batch.reward.mean(),
            }

            # Return embeddings and critic hidden for geometric metrics
            # (only last rollout's data - subsample for efficiency)
            geom_data = {
                "embeddings": traj_batch.embedding,  # (num_steps, num_envs, embed_dim)
                "critic_hidden": traj_batch.critic_hidden,  # (num_steps, num_envs, hidden_dim)
                "dones": traj_batch.done,
            }

            return (train_state, env_state, last_obs, rng), (metrics, geom_data)

        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_rng)
        runner_state, (metrics, geom_data) = jax.lax.scan(
            _update_step, runner_state, None, config["num_updates"]
        )

        return {"runner_state": runner_state, "metrics": metrics, "geom_data": geom_data}

    return train


def run_experiment(config, seed=0):
    """Run a single experiment with geometric metrics."""
    rng = jax.random.PRNGKey(seed)

    print(f"Creating training function...", flush=True)
    train_fn = make_train(config)
    print(f"JIT compiling...", flush=True)
    train_jit = jax.jit(train_fn)

    start_time = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out)
    elapsed = time.time() - start_time

    # Extract metrics
    final_metrics = jax.tree_util.tree_map(lambda x: float(x[-1]), out["metrics"])
    mean_metrics = jax.tree_util.tree_map(lambda x: float(x.mean()), out["metrics"])

    # Compute geometric metrics at sampled update points
    geom_data = out["geom_data"]
    num_updates = config["num_updates"]
    batch_size = config["num_envs"] * config["num_steps"]

    # Sample geometric metrics at 20 evenly spaced points
    sample_points = np.linspace(0, num_updates - 1, min(20, num_updates), dtype=int)
    geom_results = []

    for idx in sample_points:
        embeddings = np.array(geom_data["embeddings"][idx])  # (num_steps, num_envs, embed_dim)
        gm = compute_geometric_metrics_jax(embeddings, num_envs=config["num_envs"])
        gm["update_idx"] = int(idx)
        gm["timestep"] = int((idx + 1) * batch_size)
        geom_results.append(gm)

    # Also compute value rank from critic hidden
    for i, idx in enumerate(sample_points):
        critic_h = np.array(geom_data["critic_hidden"][idx])  # (num_steps, num_envs, hidden_dim)
        ch_flat = critic_h.reshape(-1, critic_h.shape[-1])
        if ch_flat.shape[0] > 4096:
            sub_idx = np.random.choice(ch_flat.shape[0], 4096, replace=False)
            ch_flat = ch_flat[sub_idx]
        vr = effective_rank(ch_flat)
        geom_results[i]["geom/value_rank"] = vr
        geom_results[i]["geom/value_rank_normalized"] = vr / critic_h.shape[-1]

    print(f"Training completed in {elapsed:.2f}s", flush=True)
    print(f"Final reward: {final_metrics['mean_reward']:.4f}", flush=True)

    return {
        "final_metrics": final_metrics,
        "mean_metrics": mean_metrics,
        "elapsed_time": elapsed,
        "all_metrics": out["metrics"],
        "geom_results": geom_results,
    }


def run_single(config, seed, wandb_entity="tashapais"):
    """Run and log a single experiment."""
    horizon = config["num_steps"]
    lam = config["gae_lambda"]
    run_name = f"craftax_geom_h{horizon}_lam{lam}_seed{seed}"

    wandb.init(
        project="repr-collapse-marl",
        entity=wandb_entity,
        name=run_name,
        config={**config, "seed": seed},
        tags=["craftax", f"horizon{horizon}", f"lambda{lam}", f"seed{seed}", "geometric-metrics"],
        reinit=True,
    )

    result = run_experiment(config, seed=seed)

    # Log per-update metrics
    all_metrics = result["all_metrics"]
    num_updates = len(jax.tree_util.tree_leaves(all_metrics)[0])
    batch_size = config["num_envs"] * config["num_steps"]

    for step_idx in range(num_updates):
        step_metrics = jax.tree_util.tree_map(lambda x: float(x[step_idx]), all_metrics)
        step_metrics["timestep"] = (step_idx + 1) * batch_size
        wandb.log(step_metrics, step=(step_idx + 1) * batch_size)

    # Log geometric metrics
    for gm in result["geom_results"]:
        ts = gm.pop("timestep")
        gm.pop("update_idx", None)
        wandb.log(gm, step=ts)

    # Summary
    final_geom = result["geom_results"][-1] if result["geom_results"] else {}
    wandb.summary.update({
        "final_reward": result["final_metrics"]["mean_reward"],
        "final_effective_rank": final_geom.get("geom/effective_rank", 0),
        "final_expansion_ratio": final_geom.get("geom/expansion_ratio_mean", 0),
        "final_value_rank": final_geom.get("geom/value_rank", 0),
        "elapsed_time": result["elapsed_time"],
    })

    wandb.finish()
    return result


# Experiment 3 configurations
EXP3_CONFIGS = [
    {"horizon": 100, "gae_lambda": 0.95},
    {"horizon": 100, "gae_lambda": 0.99},
    {"horizon": 500, "gae_lambda": 0.95},
    {"horizon": 500, "gae_lambda": 0.99},
    {"horizon": 2000, "gae_lambda": 0.95},
    {"horizon": 2000, "gae_lambda": 0.99},
]
NUM_SEEDS = 5


def run_all_exp3(wandb_entity="tashapais"):
    """Run all Experiment 3 configurations."""
    all_results = []

    for exp_cfg in EXP3_CONFIGS:
        horizon = exp_cfg["horizon"]
        lam = exp_cfg["gae_lambda"]

        for seed in range(NUM_SEEDS):
            config = {
                "lr": 2.5e-4,
                "num_envs": 64,
                "num_steps": horizon,
                "gamma": 0.99,
                "gae_lambda": lam,
                "num_minibatches": 4,
                "update_epochs": 4,
                "clip_eps": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "anneal_lr": True,
                "hidden_dim": 256,
                "embedding_dim": 64,
            }

            batch_size = config["num_envs"] * config["num_steps"]
            # Scale total timesteps: aim for ~1M but ensure enough updates
            total_ts = max(1_000_000, batch_size * 50)
            config["num_updates"] = total_ts // batch_size

            print(f"\n{'='*60}")
            print(f"Exp 3: horizon={horizon}, lambda={lam}, seed={seed}")
            print(f"Batch size: {batch_size}, Updates: {config['num_updates']}")
            print(f"{'='*60}")

            result = run_single(config, seed=seed, wandb_entity=wandb_entity)
            result["horizon"] = horizon
            result["gae_lambda"] = lam
            result["seed"] = seed
            all_results.append(result)

    # Save results
    summary = []
    for r in all_results:
        geom_final = r["geom_results"][-1] if r["geom_results"] else {}
        summary.append({
            "horizon": r["horizon"],
            "gae_lambda": r["gae_lambda"],
            "seed": r["seed"],
            "final_reward": r["final_metrics"]["mean_reward"],
            "value_loss": r["final_metrics"]["value_loss"],
            "effective_rank_1M": geom_final.get("geom/effective_rank", 0),
            "expansion_ratio": geom_final.get("geom/expansion_ratio_mean", 0),
            "value_rank": geom_final.get("geom/value_rank", 0),
        })

    with open("results_craftax_exp3_all.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print Table 3
    print(f"\n{'='*60}")
    print("EXPERIMENT 3 RESULTS: GAE lambda vs Horizon in Craftax")
    print(f"{'='*60}")
    print(f"{'Horizon':<10} {'Lambda':<8} {'EffRank@1M':<12} {'ValLoss':<12} {'Reward':<10}")
    print("-" * 55)

    for exp_cfg in EXP3_CONFIGS:
        h, l = exp_cfg["horizon"], exp_cfg["gae_lambda"]
        matching = [s for s in summary if s["horizon"] == h and s["gae_lambda"] == l]
        if matching:
            er = np.mean([s["effective_rank_1M"] for s in matching])
            vl = np.mean([s["value_loss"] for s in matching])
            rw = np.mean([s["final_reward"] for s in matching])
            print(f"{h:<10} {l:<8} {er:<12.1f} {vl:<12.4f} {rw:<10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=128, help="Horizon / steps per rollout")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--run_all", action="store_true", help="Run all Exp 3 configurations")
    parser.add_argument("--wandb_entity", type=str, default="tashapais")
    args = parser.parse_args()

    if args.run_all:
        run_all_exp3(wandb_entity=args.wandb_entity)
        return

    config = {
        "lr": 2.5e-4,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "gamma": 0.99,
        "gae_lambda": args.gae_lambda,
        "num_minibatches": 4,
        "update_epochs": 4,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "anneal_lr": True,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
    }

    batch_size = config["num_envs"] * config["num_steps"]
    config["num_updates"] = args.total_timesteps // batch_size

    result = run_single(config, seed=args.seed, wandb_entity=args.wandb_entity)
    print(f"\nFinal reward: {result['final_metrics']['mean_reward']:.4f}")
    if result["geom_results"]:
        final_g = result["geom_results"][-1]
        print(f"Final EffRank: {final_g.get('geom/effective_rank', 0):.1f}")
        print(f"Final ExpRatio: {final_g.get('geom/expansion_ratio_mean', 0):.2f}")


if __name__ == "__main__":
    main()
