"""
Run baseline comparisons on Craftax.

Baselines implemented:
- PPO (baseline)
- PPO + Auxiliary Contrastive (ours)
- PPO + ATC (fixed temporal offset contrastive)
- PPO + ICM (Intrinsic Curiosity Module)
- PPO + RND (Random Network Distillation)
"""

import argparse
import time
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
    """Actor-Critic network with embedding output for auxiliary losses."""
    action_dim: int
    hidden_dim: int = 256
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # Flatten observation
        x = x.reshape((x.shape[0], -1)) if len(x.shape) > 2 else x.reshape((-1,)) if len(x.shape) == 1 else x

        # Shared encoder
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        # Embedding for auxiliary losses
        embedding = nn.Dense(self.embedding_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

        # Actor head
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

        # Critic head
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

        return logits, jnp.squeeze(value, axis=-1), embedding


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    hidden_dim: int = 128
    output_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0))(x)
        return x


class ForwardModel(nn.Module):
    """Forward dynamics model for ICM."""
    hidden_dim: int = 256
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, embedding, action_onehot):
        x = jnp.concatenate([embedding, action_onehot], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(self.embedding_dim, kernel_init=orthogonal(1.0))(x)
        return x


class InverseModel(nn.Module):
    """Inverse dynamics model for ICM."""
    hidden_dim: int = 256
    action_dim: int = 17

    @nn.compact
    def __call__(self, embedding, next_embedding):
        x = jnp.concatenate([embedding, next_embedding], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        return x


class RNDNetwork(nn.Module):
    """Network for RND (used for both target and predictor)."""
    hidden_dim: int = 256
    output_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1)) if len(x.shape) > 2 else x
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0))(x)
        return x


class GeometricContrastiveLoss:
    """Our contrastive loss with geometric temporal sampling."""

    def __init__(self, temperature: float = 0.1, geometric_p: float = 0.1):
        self.temperature = temperature
        self.geometric_p = geometric_p

    def __call__(self, embeddings: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        batch_size, time_steps, embed_dim = embeddings.shape

        # Sample geometric offsets for positive pairs
        rng, offset_rng = jax.random.split(rng)
        offsets = jax.random.geometric(offset_rng, self.geometric_p, shape=(batch_size,))
        offsets = jnp.clip(offsets, 1, time_steps - 1)

        # Get anchor and positive embeddings
        anchor_indices = jnp.zeros(batch_size, dtype=jnp.int32)
        positive_indices = jnp.minimum(offsets, time_steps - 1).astype(jnp.int32)

        anchors = embeddings[jnp.arange(batch_size), anchor_indices]
        positives = embeddings[jnp.arange(batch_size), positive_indices]

        # Normalize
        anchors = anchors / (jnp.linalg.norm(anchors, axis=-1, keepdims=True) + 1e-8)
        positives = positives / (jnp.linalg.norm(positives, axis=-1, keepdims=True) + 1e-8)

        # All embeddings as negatives
        all_embeds = embeddings.reshape(-1, embed_dim)
        all_embeds = all_embeds / (jnp.linalg.norm(all_embeds, axis=-1, keepdims=True) + 1e-8)

        # Compute similarities
        pos_sim = jnp.sum(anchors * positives, axis=-1) / self.temperature
        neg_sim = jnp.matmul(anchors, all_embeds.T) / self.temperature

        # InfoNCE loss
        logsumexp_neg = jax.scipy.special.logsumexp(neg_sim, axis=-1)
        loss = -pos_sim + logsumexp_neg

        return jnp.mean(loss)


class ATCLoss:
    """ATC: Fixed temporal offset contrastive loss."""

    def __init__(self, temperature: float = 0.1, temporal_offset: int = 3):
        self.temperature = temperature
        self.temporal_offset = temporal_offset

    def __call__(self, embeddings: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        batch_size, time_steps, embed_dim = embeddings.shape

        # Fixed offset (unlike our geometric sampling)
        offset = min(self.temporal_offset, time_steps - 1)

        anchors = embeddings[:, 0]
        positives = embeddings[:, offset]

        # Normalize
        anchors = anchors / (jnp.linalg.norm(anchors, axis=-1, keepdims=True) + 1e-8)
        positives = positives / (jnp.linalg.norm(positives, axis=-1, keepdims=True) + 1e-8)

        # All embeddings as negatives
        all_embeds = embeddings.reshape(-1, embed_dim)
        all_embeds = all_embeds / (jnp.linalg.norm(all_embeds, axis=-1, keepdims=True) + 1e-8)

        # Compute similarities
        pos_sim = jnp.sum(anchors * positives, axis=-1) / self.temperature
        neg_sim = jnp.matmul(anchors, all_embeds.T) / self.temperature

        # InfoNCE loss
        logsumexp_neg = jax.scipy.special.logsumexp(neg_sim, axis=-1)
        loss = -pos_sim + logsumexp_neg

        return jnp.mean(loss)


def compute_icm_loss(
    forward_model_params,
    inverse_model_params,
    forward_model,
    inverse_model,
    embeddings: jnp.ndarray,
    next_embeddings: jnp.ndarray,
    actions: jnp.ndarray,
    action_dim: int,
    forward_coef: float = 0.2,
    inverse_coef: float = 0.8,
) -> jnp.ndarray:
    """Compute ICM loss (forward + inverse dynamics)."""
    # One-hot encode actions
    actions_onehot = jax.nn.one_hot(actions, action_dim)

    # Forward model: predict next embedding from current embedding + action
    pred_next = forward_model.apply(forward_model_params, embeddings, actions_onehot)
    forward_loss = jnp.mean(jnp.sum((pred_next - next_embeddings) ** 2, axis=-1))

    # Inverse model: predict action from current and next embedding
    pred_action_logits = inverse_model.apply(inverse_model_params, embeddings, next_embeddings)
    inverse_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(pred_action_logits, actions)
    )

    return forward_coef * forward_loss + inverse_coef * inverse_loss


def compute_rnd_loss(
    predictor_params,
    target_params,
    predictor_network,
    target_network,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute RND loss (predictor tries to match fixed random target)."""
    # Target network is fixed (random)
    target_features = target_network.apply(target_params, obs)
    # Predictor network is trained
    pred_features = predictor_network.apply(predictor_params, obs)

    # MSE loss
    loss = jnp.mean(jnp.sum((pred_features - target_features) ** 2, axis=-1))
    return loss


def make_train(config):
    """Create training function for specified method."""

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["num_updates"]
        return config["lr"] * frac

    def train(rng):
        # Initialize network
        rng, init_rng = jax.random.split(rng)
        network = ActorCritic(
            action_dim=env.action_space(env_params).n,
            hidden_dim=config["hidden_dim"],
            embedding_dim=config["embedding_dim"],
        )

        dummy_obs = jnp.zeros((1, env.observation_space(env_params).shape[0]))
        network_params = network.init(init_rng, dummy_obs)

        # Initialize projection head for contrastive methods
        proj_head = ProjectionHead(hidden_dim=128, output_dim=config["embedding_dim"])
        rng, proj_rng = jax.random.split(rng)
        proj_params = proj_head.init(proj_rng, jnp.zeros((1, config["embedding_dim"])))

        # Initialize ICM models if needed
        forward_model = ForwardModel(hidden_dim=256, embedding_dim=config["embedding_dim"])
        inverse_model = InverseModel(hidden_dim=256, action_dim=env.action_space(env_params).n)
        rng, icm_rng = jax.random.split(rng)
        icm_rng1, icm_rng2 = jax.random.split(icm_rng)
        forward_params = forward_model.init(
            icm_rng1,
            jnp.zeros((1, config["embedding_dim"])),
            jnp.zeros((1, env.action_space(env_params).n))
        )
        inverse_params = inverse_model.init(
            icm_rng2,
            jnp.zeros((1, config["embedding_dim"])),
            jnp.zeros((1, config["embedding_dim"]))
        )

        # Initialize RND networks if needed
        rnd_target = RNDNetwork(hidden_dim=256, output_dim=64)
        rnd_predictor = RNDNetwork(hidden_dim=256, output_dim=64)
        rng, rnd_rng = jax.random.split(rng)
        rnd_rng1, rnd_rng2 = jax.random.split(rnd_rng)
        rnd_target_params = rnd_target.init(rnd_rng1, dummy_obs)
        rnd_predictor_params = rnd_predictor.init(rnd_rng2, dummy_obs)

        # Combine all params
        all_params = {
            "network": network_params,
            "proj_head": proj_params,
            "forward_model": forward_params,
            "inverse_model": inverse_params,
            "rnd_predictor": rnd_predictor_params,
            "rnd_target": rnd_target_params,  # Fixed, not trained
        }

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

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=all_params,
            tx=tx,
        )

        # Initialize environment
        rng, env_rng = jax.random.split(rng)
        env_rng = jax.random.split(env_rng, config["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(env_rng, env_params)

        # Contrastive loss functions
        geometric_contrastive = GeometricContrastiveLoss(
            temperature=config["contrastive_temperature"],
            geometric_p=1 - config["contrastive_gamma"],
        )
        atc_contrastive = ATCLoss(
            temperature=config["contrastive_temperature"],
            temporal_offset=3,
        )

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state

            # Select action
            rng, action_rng = jax.random.split(rng)
            logits, value, embedding = network.apply(train_state.params["network"], last_obs)
            action = jax.random.categorical(action_rng, logits)
            log_prob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]

            # Step environment
            rng, step_rng = jax.random.split(rng)
            step_rng = jax.random.split(step_rng, config["num_envs"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_rng, env_state, action, env_params
            )

            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=last_obs,
                next_obs=obsv,
                embedding=embedding,
            )

            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state

            # Collect trajectory
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["num_steps"]
            )

            # Compute advantages
            train_state, env_state, last_obs, rng = runner_state
            _, last_val, _ = network.apply(train_state.params["network"], last_obs)

            def _compute_gae(carry, transition):
                gae, next_value = carry
                delta = transition.reward + config["gamma"] * next_value * (1 - transition.done) - transition.value
                gae = delta + config["gamma"] * config["gae_lambda"] * (1 - transition.done) * gae
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(
                _compute_gae, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True
            )
            returns = advantages + traj_batch.value

            def _update_epoch(carry, unused):
                train_state, rng = carry

                # Shuffle and create minibatches
                rng, perm_rng = jax.random.split(rng)
                batch_size = config["num_envs"] * config["num_steps"]
                permutation = jax.random.permutation(perm_rng, batch_size)

                batch = (traj_batch, advantages, returns)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)

                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["num_minibatches"], -1) + x.shape[1:]),
                    shuffled_batch
                )

                def _update_minibatch(train_state, minibatch):
                    traj, mb_advantages, mb_returns = minibatch

                    def _loss_fn(params):
                        # PPO losses
                        logits, value, embedding = network.apply(params["network"], traj.obs)
                        log_prob = jax.nn.log_softmax(logits)[jnp.arange(traj.action.shape[0]), traj.action]

                        # Value loss
                        value_loss = 0.5 * jnp.mean((value - mb_returns) ** 2)

                        # Policy loss with clipping
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        mb_advantages_normalized = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        pg_loss1 = -mb_advantages_normalized * ratio
                        pg_loss2 = -mb_advantages_normalized * jnp.clip(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"])
                        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

                        # Entropy bonus
                        entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1))

                        # Auxiliary loss based on method
                        aux_loss = 0.0

                        if config["method"] == "contrastive":
                            # Our method: geometric temporal contrastive
                            # Reshape embeddings for contrastive loss
                            embed_for_contrast = embedding.reshape(config["num_envs"], -1, config["embedding_dim"])
                            projected = proj_head.apply(params["proj_head"], embed_for_contrast)
                            rng_contrast = jax.random.PRNGKey(0)  # Deterministic for minibatch
                            aux_loss = geometric_contrastive(projected, rng_contrast)

                        elif config["method"] == "atc":
                            # ATC: fixed temporal offset contrastive
                            embed_for_contrast = embedding.reshape(config["num_envs"], -1, config["embedding_dim"])
                            projected = proj_head.apply(params["proj_head"], embed_for_contrast)
                            rng_contrast = jax.random.PRNGKey(0)
                            aux_loss = atc_contrastive(projected, rng_contrast)

                        elif config["method"] == "icm":
                            # ICM: forward + inverse dynamics
                            _, _, next_embedding = network.apply(params["network"], traj.next_obs)
                            aux_loss = compute_icm_loss(
                                params["forward_model"],
                                params["inverse_model"],
                                forward_model,
                                inverse_model,
                                embedding,
                                next_embedding,
                                traj.action,
                                env.action_space(env_params).n,
                            )

                        elif config["method"] == "rnd":
                            # RND: random network distillation
                            aux_loss = compute_rnd_loss(
                                params["rnd_predictor"],
                                params["rnd_target"],
                                rnd_predictor,
                                rnd_target,
                                traj.obs,
                            )

                        total_loss = pg_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
                        total_loss += config["aux_coef"] * aux_loss

                        return total_loss, (pg_loss, value_loss, entropy, aux_loss)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, aux), grads = grad_fn(train_state.params)

                    # Don't update RND target (it's fixed) - zero out its gradients
                    grads = {
                        k: (jax.tree_util.tree_map(jnp.zeros_like, v) if k == "rnd_target" else v)
                        for k, v in grads.items()
                    }

                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, aux)

                train_state, losses = jax.lax.scan(_update_minibatch, train_state, minibatches)
                return (train_state, rng), losses

            (train_state, rng), losses = jax.lax.scan(
                _update_epoch, (train_state, rng), None, config["update_epochs"]
            )

            metrics = {
                "total_loss": losses[0].mean(),
                "pg_loss": losses[1][0].mean(),
                "value_loss": losses[1][1].mean(),
                "entropy": losses[1][2].mean(),
                "aux_loss": losses[1][3].mean(),
                "mean_reward": traj_batch.reward.mean(),
            }

            return (train_state, env_state, last_obs, rng), metrics

        # Run training
        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["num_updates"])

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def run_experiment(config, seed=0):
    """Run a single experiment."""
    rng = jax.random.PRNGKey(seed)

    print(f"Creating training function for method: {config['method']}...", flush=True)
    train_fn = make_train(config)
    print(f"JIT compiling...", flush=True)
    train_jit = jax.jit(train_fn)

    print(f"Starting training with method={config['method']}, aux_coef={config['aux_coef']}", flush=True)
    start_time = time.time()

    out = train_jit(rng)

    print("Waiting for computation to complete...", flush=True)
    jax.block_until_ready(out)

    elapsed = time.time() - start_time

    final_metrics = jax.tree_util.tree_map(lambda x: float(x[-1]), out["metrics"])
    mean_metrics = jax.tree_util.tree_map(lambda x: float(x.mean()), out["metrics"])

    print(f"Training completed in {elapsed:.2f}s", flush=True)
    print(f"Final reward: {final_metrics['mean_reward']:.4f}", flush=True)
    print(f"Mean reward: {mean_metrics['mean_reward']:.4f}", flush=True)
    print(f"Final aux_loss: {final_metrics['aux_loss']:.4f}", flush=True)

    return {
        "final_metrics": final_metrics,
        "mean_metrics": mean_metrics,
        "elapsed_time": elapsed,
        "all_metrics": out["metrics"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="ppo",
                        choices=["ppo", "contrastive", "atc", "icm", "rnd"],
                        help="Training method")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--aux_coef", type=float, default=0.001,
                        help="Coefficient for auxiliary loss")
    args = parser.parse_args()

    config = {
        "method": args.method,
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
        "aux_coef": args.aux_coef if args.method != "ppo" else 0.0,
        "contrastive_temperature": 0.1,
        "contrastive_gamma": 0.977,
    }

    batch_size = config["num_envs"] * config["num_steps"]
    config["num_updates"] = args.total_timesteps // batch_size

    method_names = {
        "ppo": "PPO Baseline",
        "contrastive": "PPO + Geometric Contrastive (Ours)",
        "atc": "PPO + ATC (Fixed Temporal)",
        "icm": "PPO + ICM",
        "rnd": "PPO + RND",
    }

    print("=" * 60, flush=True)
    print(method_names[args.method], flush=True)
    print("=" * 60, flush=True)
    print(f"Total timesteps: {args.total_timesteps:,}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Num updates: {config['num_updates']}", flush=True)
    print(f"Aux coefficient: {config['aux_coef']}", flush=True)
    print("=" * 60, flush=True)

    all_results = []
    for seed in range(args.num_seeds):
        print(f"\n--- Seed {seed} ---", flush=True)
        result = run_experiment(config, seed=seed)
        all_results.append(result)

    mean_rewards = [r["mean_metrics"]["mean_reward"] for r in all_results]
    final_rewards = [r["final_metrics"]["mean_reward"] for r in all_results]

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Method: {method_names[args.method]}", flush=True)
    print(f"Mean reward across seeds: {np.mean(mean_rewards):.4f} +/- {np.std(mean_rewards):.4f}", flush=True)
    print(f"Final reward across seeds: {np.mean(final_rewards):.4f} +/- {np.std(final_rewards):.4f}", flush=True)

    return all_results


if __name__ == "__main__":
    main()
