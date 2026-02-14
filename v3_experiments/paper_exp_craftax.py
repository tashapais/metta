"""
Paper experiments: Craftax (GPU 2 & 3)
Computes geometric metrics for Tables 3, 4, 6, and ablations (Tables depth, batch).
"""
import argparse
import json
import time
import os

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import numpy as np
from typing import NamedTuple
import wandb

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG


def compute_effective_rank_np(Z: np.ndarray) -> float:
    """Compute effective rank via SVD entropy (numpy version)."""
    if Z.shape[0] < 2:
        return 1.0
    try:
        S = np.linalg.svd(Z.astype(np.float32), compute_uv=False)
        S = S[S > 1e-10]
        if len(S) == 0:
            return 1.0
        S_norm = S / S.sum()
        entropy = -(S_norm * np.log(S_norm + 1e-12)).sum()
        return float(np.exp(entropy))
    except Exception:
        return 1.0


def compute_svd_ratio_np(Z: np.ndarray) -> float:
    """sigma_1 / sigma_10."""
    try:
        S = np.linalg.svd(Z.astype(np.float32), compute_uv=False)
        if len(S) >= 10:
            return float(S[0] / (S[9] + 1e-10))
        return float(S[0] / (S[-1] + 1e-10))
    except Exception:
        return 1.0


def compute_expansion_ratio_np(embeddings: np.ndarray) -> float:
    """var(late) / var(early)."""
    T = embeddings.shape[0]
    if T < 5:
        return 1.0
    early_cutoff = max(1, int(T * 0.2))
    late_cutoff = max(early_cutoff + 1, int(T * 0.8))
    early = embeddings[:early_cutoff].reshape(-1, embeddings.shape[-1])
    late = embeddings[late_cutoff:].reshape(-1, embeddings.shape[-1])
    if early.shape[0] < 2 or late.shape[0] < 2:
        return 1.0
    early_var = early.var(axis=0).sum()
    late_var = late.var(axis=0).sum()
    return float(late_var / (early_var + 1e-10))


class ActorCritic(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    embedding_dim: int = 64
    num_encoder_layers: int = 2

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1)).astype(jnp.float32)
        # Variable depth encoder
        for _ in range(self.num_encoder_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        embedding = nn.Dense(self.embedding_dim)(x)
        actor = nn.Dense(self.hidden_dim)(x)
        actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim)(actor)
        critic = nn.Dense(self.hidden_dim)(x)
        critic = nn.relu(critic)
        value = nn.Dense(1)(critic)
        return logits, value.squeeze(-1), embedding


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    embedding: jnp.ndarray


def make_train(config):
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
            num_encoder_layers=config.get("num_encoder_layers", 2),
        )
        dummy_obs = jnp.zeros((1,) + env.observation_space(env_params).shape)
        params = network.init(init_rng, dummy_obs)

        if config["anneal_lr"]:
            tx = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]),
                             optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]),
                             optax.adam(config["lr"], eps=1e-5))

        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, config["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state

            def _env_step(carry, unused):
                train_state, env_state, last_obs, rng = carry
                rng, action_rng, step_rng = jax.random.split(rng, 3)
                logits, value, embedding = network.apply(train_state.params, last_obs)
                action_dist = jax.nn.softmax(logits)
                action = jax.random.categorical(action_rng, logits)
                log_prob = jnp.log(action_dist[jnp.arange(config["num_envs"]), action] + 1e-8)
                step_rng = jax.random.split(step_rng, config["num_envs"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    step_rng, env_state, action, env_params)
                transition = Transition(done=done, action=action, value=value, reward=reward,
                                        log_prob=log_prob, obs=last_obs, embedding=embedding)
                return (train_state, env_state, obsv, rng), transition

            runner_state_inner, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, config["num_steps"])
            train_state, env_state, last_obs, rng = runner_state_inner

            _, last_val, _ = network.apply(train_state.params, last_obs)

            def _calculate_gae(carry, transition):
                gae, next_value = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["gamma"] * next_value * (1 - done) - value
                gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(_calculate_gae, (jnp.zeros(config["num_envs"]), last_val),
                                         traj_batch, reverse=True)
            returns = advantages + traj_batch.value

            def _update_epoch(carry, unused):
                train_state, rng = carry
                rng, perm_rng = jax.random.split(rng)
                batch_size = config["num_steps"] * config["num_envs"]
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch)
                advantages_flat = advantages.reshape(batch_size)
                returns_flat = returns.reshape(batch_size)
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = jax.tree_util.tree_map(lambda x: x[permutation], batch)
                advantages_flat = advantages_flat[permutation]
                returns_flat = returns_flat[permutation]

                def _update_minibatch(train_state, batch_info):
                    mb_batch, mb_advantages, mb_returns = batch_info
                    def loss_fn(params):
                        logits, value, embedding = network.apply(params, mb_batch.obs)
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
                    _update_minibatch, train_state, (minibatches, mb_advantages, mb_returns))
                return (train_state, rng), losses

            (train_state, rng), losses = jax.lax.scan(
                _update_epoch, (train_state, rng), None, config["update_epochs"])

            metrics = {
                "total_loss": losses[0].mean(),
                "pg_loss": losses[1][0].mean(),
                "value_loss": losses[1][1].mean(),
                "entropy": losses[1][2].mean(),
                "mean_reward": traj_batch.reward.mean(),
                "embeddings_sample": traj_batch.embedding,  # (num_steps, num_envs, embed_dim)
            }
            return (train_state, env_state, last_obs, rng), metrics

        runner_state = (train_state, env_state, obsv, rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["num_updates"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def record_craftax_video(params, network, config, seed=0, max_steps=500):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(seed + 1000)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)
    frames = []
    for _ in range(max_steps):
        pixels = render_craftax_pixels(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG)
        frame = np.array(pixels).astype(np.uint8)
        frames.append(frame)
        rng, action_rng = jax.random.split(rng)
        obs_batch = obs[None, ...]
        logits, _, _ = network.apply(params, obs_batch)
        action = jax.random.categorical(action_rng, logits[0])
        rng, step_rng = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)
        if done:
            break
    video = np.stack(frames).transpose(0, 3, 1, 2)
    return video


def run_single_experiment(config, seed=0, record_video=True):
    """Run one Craftax experiment and return geometric metrics."""
    gpu_id = config.get("gpu", 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # JAX will use first visible GPU
    jax_device = jax.devices()[0]
    print(f"[Craftax h={config.get('horizon', 'default')} lam={config['gae_lambda']}] Using: {jax_device}")

    rng = jax.random.PRNGKey(seed)

    run_name = f"paper_craftax_h{config.get('horizon', 'default')}_lam{config['gae_lambda']}_d{config.get('num_encoder_layers', 2)}_b{config.get('actual_batch_size', 'default')}_seed{seed}"
    wandb.init(project="representation-collapse", name=run_name, config={**config, "seed": seed}, reinit=True)

    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    start_time = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out)
    elapsed = time.time() - start_time

    metrics = out["metrics"]
    num_updates = config["num_updates"]

    # Extract embedding samples for geometric metrics
    # metrics["embeddings_sample"] is (num_updates, num_steps, num_envs, embed_dim)
    all_embeddings = np.array(metrics["embeddings_sample"])
    rewards_over_time = np.array(metrics["mean_reward"])

    # Compute geometric metrics at end of training
    final_embs = all_embeddings[-1].reshape(-1, config["embedding_dim"])
    sample_size = min(1024, final_embs.shape[0])
    idx = np.random.choice(final_embs.shape[0], sample_size, replace=False)
    final_eff_rank = compute_effective_rank_np(final_embs[idx])
    final_svd_ratio = compute_svd_ratio_np(final_embs[idx])

    # Compute effective rank at 1M timesteps (approximately)
    batch_size = config["num_envs"] * config["num_steps"]
    target_update = min(1_000_000 // batch_size, num_updates - 1)
    mid_embs = all_embeddings[target_update].reshape(-1, config["embedding_dim"])
    idx_mid = np.random.choice(mid_embs.shape[0], min(1024, mid_embs.shape[0]), replace=False)
    eff_rank_at_1m = compute_effective_rank_np(mid_embs[idx_mid])

    # Expansion ratio (over entire training trajectory of embeddings)
    # Use last update's trajectory
    last_traj_embs = all_embeddings[-1]  # (num_steps, num_envs, embed_dim)
    expansion_ratios = []
    for env_idx in range(min(config["num_envs"], 16)):
        ep_embs = last_traj_embs[:, env_idx, :]
        exp_r = compute_expansion_ratio_np(ep_embs)
        expansion_ratios.append(exp_r)
    mean_exp_ratio = np.mean(expansion_ratios)

    # Value loss at end
    final_value_loss = float(metrics["value_loss"][-1])
    final_reward = float(metrics["mean_reward"][-1])

    # Effective rank history for lead time analysis
    eff_rank_history = []
    for i in range(0, num_updates, max(1, num_updates // 50)):
        embs = all_embeddings[i].reshape(-1, config["embedding_dim"])
        idx_h = np.random.choice(embs.shape[0], min(512, embs.shape[0]), replace=False)
        er = compute_effective_rank_np(embs[idx_h])
        eff_rank_history.append((i * batch_size, er))

    # Log per-update metrics to wandb
    for step_idx in range(0, num_updates, max(1, num_updates // 200)):
        step_metrics = jax.tree_util.tree_map(lambda x: float(x[step_idx]),
                                               {k: v for k, v in metrics.items() if k != "embeddings_sample"})
        step_metrics["timestep"] = (step_idx + 1) * batch_size

        # Add geometric metrics periodically
        if step_idx % max(1, num_updates // 50) == 0:
            embs = all_embeddings[step_idx].reshape(-1, config["embedding_dim"])
            idx_s = np.random.choice(embs.shape[0], min(512, embs.shape[0]), replace=False)
            step_metrics["geometric/effective_rank"] = compute_effective_rank_np(embs[idx_s])
            step_metrics["geometric/svd_ratio"] = compute_svd_ratio_np(embs[idx_s])

        wandb.log(step_metrics, step=(step_idx + 1) * batch_size)

    # Record and log video
    if record_video:
        try:
            env_for_video = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
            action_dim = env_for_video.action_space(env_for_video.default_params).n
            network = ActorCritic(
                action_dim=action_dim,
                hidden_dim=config["hidden_dim"],
                embedding_dim=config["embedding_dim"],
                num_encoder_layers=config.get("num_encoder_layers", 2),
            )
            train_state = out["runner_state"][0]
            video = record_craftax_video(train_state.params, network, config, seed=seed)
            if video is not None:
                wandb.log({"video/episode": wandb.Video(video, fps=15, format="mp4")})
                print("Video logged to wandb")
        except Exception as e:
            print(f"Video recording failed: {e}")

    wandb.finish()

    result = {
        "horizon": config.get("horizon", "default"),
        "gae_lambda": config["gae_lambda"],
        "num_encoder_layers": config.get("num_encoder_layers", 2),
        "batch_size_actual": config.get("actual_batch_size", config["num_envs"] * config["num_steps"]),
        "eff_rank_at_1m": round(eff_rank_at_1m, 1),
        "final_eff_rank": round(final_eff_rank, 1),
        "final_svd_ratio": round(final_svd_ratio, 1),
        "expansion_ratio": round(mean_exp_ratio, 2),
        "value_loss": round(final_value_loss, 4),
        "final_reward": round(final_reward, 4),
        "elapsed_time": round(elapsed, 1),
        # Threshold checks
        "effrank_below_0.3d": bool(eff_rank_at_1m < 0.3 * config["embedding_dim"]),
        "exp_ratio_below_1": bool(mean_exp_ratio < 1.0),
        "svd_ratio_above_10": bool(final_svd_ratio > 10.0),
    }

    print(f"Result: {json.dumps(result, indent=2)}")
    return result


def aggregate_seed_results(seed_results):
    """Aggregate results across seeds: compute mean ± std for numeric fields."""
    if not seed_results:
        return {}
    aggregated = {}
    keys = seed_results[0].keys()
    for key in keys:
        values = [r[key] for r in seed_results]
        if isinstance(values[0], (int, float)):
            arr = np.array(values, dtype=float)
            aggregated[key] = round(float(arr.mean()), 3)
            aggregated[f"{key}_std"] = round(float(arr.std()), 3)
            aggregated[f"{key}_seeds"] = [round(float(v), 3) for v in values]
        elif isinstance(values[0], bool):
            aggregated[key] = bool(np.mean(values) > 0.5)
        else:
            aggregated[key] = values[0]
    return aggregated


def run_experiment_with_seeds(config, num_seeds, exp_label, record_video=True):
    """Run a single experiment config across multiple seeds and aggregate."""
    print(f"\n{'='*60}")
    print(f"Craftax: {exp_label} ({num_seeds} seeds)")
    print(f"{'='*60}")

    seed_results = []
    for seed in range(num_seeds):
        print(f"\n--- Seed {seed}/{num_seeds} ---")
        # Only record video for first seed
        result = run_single_experiment(config, seed=seed, record_video=(record_video and seed == 0))
        seed_results.append(result)

    aggregated = aggregate_seed_results(seed_results)
    print(f"\nAggregated ({exp_label}): {json.dumps(aggregated, indent=2)}")
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--part", type=int, default=1, choices=[1, 2],
                        help="Part 1: horizon experiments. Part 2: ablations")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_seeds", type=int, default=5)
    args = parser.parse_args()

    base_config = {
        "lr": 2.5e-4,
        "num_envs": 64,
        "num_steps": 128,
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
        "gpu": args.gpu,
    }

    all_results = []

    if args.part == 1:
        experiments = [
            {"horizon": 100, "gae_lambda": 0.95, "num_steps": 100},
            {"horizon": 100, "gae_lambda": 0.99, "num_steps": 100},
            {"horizon": 500, "gae_lambda": 0.95, "num_steps": 128},
        ]
        for exp in experiments:
            config = {**base_config, **exp}
            batch_size = config["num_envs"] * config["num_steps"]
            config["num_updates"] = args.total_timesteps // batch_size
            config["actual_batch_size"] = batch_size
            config["total_timesteps"] = args.total_timesteps

            label = f"horizon={exp['horizon']}, lambda={exp['gae_lambda']}"
            aggregated = run_experiment_with_seeds(config, args.num_seeds, label)
            all_results.append(aggregated)

    elif args.part == 2:
        # Table 3: Horizon x Lambda (last 3 rows)
        experiments = [
            {"horizon": 500, "gae_lambda": 0.99, "num_steps": 128},
            {"horizon": 2000, "gae_lambda": 0.95, "num_steps": 128},
            {"horizon": 2000, "gae_lambda": 0.99, "num_steps": 128},
        ]
        for exp in experiments:
            config = {**base_config, **exp}
            batch_size = config["num_envs"] * config["num_steps"]
            config["num_updates"] = args.total_timesteps // batch_size
            config["actual_batch_size"] = batch_size
            config["total_timesteps"] = args.total_timesteps

            label = f"horizon={exp['horizon']}, lambda={exp['gae_lambda']}"
            aggregated = run_experiment_with_seeds(config, args.num_seeds, label)
            all_results.append(aggregated)

        # Table depth: Encoder depth ablation
        for depth in [2, 4, 8]:
            config = {**base_config}
            config["num_encoder_layers"] = depth
            config["gae_lambda"] = 0.95
            config["horizon"] = "default"
            batch_size = config["num_envs"] * config["num_steps"]
            config["num_updates"] = args.total_timesteps // batch_size
            config["actual_batch_size"] = batch_size
            config["total_timesteps"] = args.total_timesteps

            label = f"depth={depth} layers"
            aggregated = run_experiment_with_seeds(config, args.num_seeds, label, record_video=False)
            all_results.append(aggregated)

        # Table batch: Batch size ablation
        for bs_envs in [16, 64, 256]:
            config = {**base_config}
            config["num_envs"] = bs_envs
            config["gae_lambda"] = 0.95
            config["horizon"] = "default"
            batch_size = config["num_envs"] * config["num_steps"]
            config["num_updates"] = min(args.total_timesteps // batch_size, 500)
            config["actual_batch_size"] = batch_size
            config["total_timesteps"] = args.total_timesteps

            label = f"batch_size={batch_size} (num_envs={bs_envs})"
            aggregated = run_experiment_with_seeds(config, args.num_seeds, label, record_video=False)
            all_results.append(aggregated)

    output_path = f"/home/ubuntu/metta/v3_experiments/results_craftax_part{args.part}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
