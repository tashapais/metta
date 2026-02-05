"""
Craftax experiments for contrastive learning paper.
Runs PPO baseline and PPO+Contrastive on Craftax-Symbolic.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import numpy as np
from typing import Sequence, NamedTuple, Any
from functools import partial
import time
import argparse

# Craftax imports
from craftax.craftax_env import make_craftax_env_from_name


class ActorCritic(nn.Module):
    """Simple actor-critic network."""
    action_dim: int
    hidden_dim: int = 256
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # Flatten observation
        x = x.reshape((x.shape[0], -1)).astype(jnp.float32)

        # Shared encoder
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Embedding for contrastive learning
        embedding = nn.Dense(self.embedding_dim)(x)

        # Actor head
        actor = nn.Dense(self.hidden_dim)(x)
        actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim)(actor)

        # Critic head
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


class ContrastiveLoss:
    """InfoNCE contrastive loss with geometric temporal sampling."""

    def __init__(self, temperature: float = 0.1, geometric_p: float = 0.1):
        self.temperature = temperature
        self.geometric_p = geometric_p  # p parameter for geometric distribution

    def __call__(self, embeddings: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Compute InfoNCE loss over batch of embeddings.
        embeddings: (batch_size, time_steps, embedding_dim)
        """
        batch_size, time_steps, embed_dim = embeddings.shape

        if time_steps < 2:
            return jnp.array(0.0)

        # Sample geometric offsets for positive pairs
        key, subkey = jax.random.split(key)
        offsets = jax.random.geometric(subkey, self.geometric_p, shape=(batch_size,))
        offsets = jnp.clip(offsets, 1, time_steps - 1)

        # Get anchor and positive embeddings
        anchor_idx = jnp.zeros(batch_size, dtype=jnp.int32)
        positive_idx = jnp.minimum(anchor_idx + offsets, time_steps - 1).astype(jnp.int32)

        anchors = embeddings[:, 0, :]  # (batch_size, embed_dim)
        positives = jax.vmap(lambda e, i: e[i])(embeddings, positive_idx)  # (batch_size, embed_dim)

        # Normalize embeddings
        anchors = anchors / (jnp.linalg.norm(anchors, axis=-1, keepdims=True) + 1e-8)
        positives = positives / (jnp.linalg.norm(positives, axis=-1, keepdims=True) + 1e-8)

        # All embeddings as negatives (flatten batch and time)
        all_embeds = embeddings.reshape(-1, embed_dim)
        all_embeds = all_embeds / (jnp.linalg.norm(all_embeds, axis=-1, keepdims=True) + 1e-8)

        # Compute similarities
        pos_sim = jnp.sum(anchors * positives, axis=-1) / self.temperature  # (batch_size,)
        neg_sim = jnp.matmul(anchors, all_embeds.T) / self.temperature  # (batch_size, batch*time)

        # InfoNCE loss
        logsumexp_neg = jax.scipy.special.logsumexp(neg_sim, axis=-1)
        loss = -pos_sim + logsumexp_neg

        return jnp.mean(loss)


class FlaxSAEncoder(nn.Module):
    """State-Action encoder for GC-CRL (Flax). 4-layer MLP with LayerNorm + SiLU."""
    hidden_dim: int = 1024
    embed_dim: int = 64

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.silu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        x = nn.silu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        x = nn.silu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        x = nn.silu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        return nn.Dense(self.embed_dim)(x)


class FlaxGEncoder(nn.Module):
    """Goal encoder for GC-CRL (Flax). 4-layer MLP with LayerNorm + SiLU."""
    hidden_dim: int = 1024
    embed_dim: int = 64

    @nn.compact
    def __call__(self, goal):
        x = nn.silu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(goal)))
        x = nn.silu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        x = nn.silu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        x = nn.silu(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        return nn.Dense(self.embed_dim)(x)


def gc_crl_loss_fn(sa_apply, g_apply, sa_params, g_params, obs, actions, action_dim, config, key):
    """Compute GC-CRL InfoNCE loss (pure function for JAX JIT).

    Args:
        sa_apply: SAEncoder apply function
        g_apply: GEncoder apply function
        sa_params: SAEncoder parameters
        g_params: GEncoder parameters
        obs: (batch_size, obs_dim) flattened observations
        actions: (batch_size,) discrete action indices
        action_dim: int, number of discrete actions
        config: dict with gc_contrastive_coef, gc_logsumexp_coef
        key: JAX PRNG key

    Returns:
        (loss, metrics_dict)
    """
    # One-hot encode discrete actions
    actions_onehot = jax.nn.one_hot(actions, action_dim).astype(jnp.float32)
    obs_flat = obs.reshape(obs.shape[0], -1).astype(jnp.float32)

    # Encode
    sa_repr = sa_apply(sa_params, obs_flat, actions_onehot)
    g_repr = g_apply(g_params, obs_flat)  # Goals = same obs (shuffled by minibatch)

    n = obs_flat.shape[0]

    # Pairwise negative Euclidean distance
    diff = sa_repr[:, None, :] - g_repr[None, :, :]  # [N, N, embed_dim]
    logits = -jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)  # [N, N]

    # InfoNCE
    diag_logits = jnp.diag(logits)
    logsumexp = jax.scipy.special.logsumexp(logits, axis=1)
    infonce_loss = -jnp.mean(diag_logits - logsumexp)

    # Logsumexp regularization
    logsumexp_reg = config["gc_logsumexp_coef"] * jnp.mean(logsumexp ** 2)

    total_loss = (infonce_loss + logsumexp_reg) * config["gc_contrastive_coef"]

    # Metrics
    I = jnp.eye(n)
    correct = (jnp.argmax(logits, axis=1) == jnp.arange(n)).astype(jnp.float32)
    categorical_accuracy = jnp.mean(correct)
    logits_pos = jnp.sum(logits * I) / jnp.sum(I)
    logits_neg = jnp.sum(logits * (1 - I)) / jnp.maximum(jnp.sum(1 - I), 1.0)

    metrics = {
        "gc_categorical_accuracy": categorical_accuracy,
        "gc_logits_pos": logits_pos,
        "gc_logits_neg": logits_neg,
        "gc_infonce_loss": infonce_loss,
        "gc_logsumexp_reg": logsumexp_reg,
    }

    return total_loss, metrics


def make_train(config):
    """Create training function."""

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    use_gc_crl = config.get("method") == "gc_crl"
    action_dim = env.action_space(env_params).n

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["num_updates"]
        return config["lr"] * frac

    # Create GC-CRL encoder modules at Python level (before JIT)
    if use_gc_crl:
        sa_encoder = FlaxSAEncoder(
            hidden_dim=config["gc_hidden_dim"],
            embed_dim=config["gc_embed_dim"],
        )
        g_encoder = FlaxGEncoder(
            hidden_dim=config["gc_hidden_dim"],
            embed_dim=config["gc_embed_dim"],
        )

    def train(rng):
        # Initialize network
        rng, init_rng = jax.random.split(rng)
        network = ActorCritic(
            action_dim=action_dim,
            hidden_dim=config["hidden_dim"],
            embedding_dim=config["embedding_dim"],
        )

        # Dummy input for initialization
        rng, obs_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1,) + env.observation_space(env_params).shape)
        params = network.init(init_rng, dummy_obs)

        # Optimizer
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

        # Contrastive loss
        contrastive_loss_fn = ContrastiveLoss(
            temperature=config["contrastive_temperature"],
            geometric_p=1.0 - config["contrastive_gamma"],
        )

        # Initialize GC-CRL encoders and train state
        if use_gc_crl:
            rng, sa_init_rng, g_init_rng = jax.random.split(rng, 3)
            obs_flat_dim = int(np.prod(env.observation_space(env_params).shape))
            dummy_state = jnp.zeros((1, obs_flat_dim))
            dummy_action_oh = jnp.zeros((1, action_dim))

            sa_params = sa_encoder.init(sa_init_rng, dummy_state, dummy_action_oh)
            g_params = g_encoder.init(g_init_rng, dummy_state)

            gc_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(config["gc_lr"], eps=1e-5),
            )
            gc_params = {"sa": sa_params, "g": g_params}
            gc_train_state = TrainState.create(
                apply_fn=sa_encoder.apply,  # placeholder
                params=gc_params,
                tx=gc_tx,
            )

        # Initialize environment
        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, config["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # Training loop
        def _update_step(runner_state, unused):
            if use_gc_crl:
                train_state, gc_train_state, env_state, last_obs, rng = runner_state
            else:
                train_state, env_state, last_obs, rng = runner_state

            # Collect rollout
            def _env_step(carry, unused):
                train_state, env_state, last_obs, rng = carry
                rng, action_rng, step_rng = jax.random.split(rng, 3)

                logits, value, embedding = network.apply(train_state.params, last_obs)
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
                )
                return (train_state, env_state, obsv, rng), transition

            runner_state_inner, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, config["num_steps"]
            )
            train_state, env_state, last_obs, rng = runner_state_inner

            # Calculate advantages (GAE)
            _, last_val, _ = network.apply(train_state.params, last_obs)

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

            if use_gc_crl:
                # GC-CRL path: separate optimizers for policy and GC encoders
                def _update_epoch(carry, unused):
                    train_state, gc_train_state, rng = carry
                    rng, perm_rng, gc_rng = jax.random.split(rng, 3)

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

                    def _update_minibatch(carry, batch_info):
                        train_state, gc_train_state = carry
                        mb_batch, mb_advantages, mb_returns = batch_info

                        # PPO loss
                        def ppo_loss_fn(params):
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

                        ppo_grad_fn = jax.value_and_grad(ppo_loss_fn, has_aux=True)
                        (ppo_loss, ppo_aux), ppo_grads = ppo_grad_fn(train_state.params)
                        train_state = train_state.apply_gradients(grads=ppo_grads)

                        # GC-CRL loss
                        def gc_loss_wrapper(gc_params):
                            return gc_crl_loss_fn(
                                sa_encoder.apply, g_encoder.apply,
                                gc_params["sa"], gc_params["g"],
                                mb_batch.obs, mb_batch.action, action_dim,
                                config, gc_rng,
                            )

                        gc_grad_fn = jax.value_and_grad(gc_loss_wrapper, has_aux=True)
                        (gc_loss, gc_metrics), gc_grads = gc_grad_fn(gc_train_state.params)
                        gc_train_state = gc_train_state.apply_gradients(grads=gc_grads)

                        combined_loss = ppo_loss + gc_loss
                        return (train_state, gc_train_state), (combined_loss, (ppo_aux[0], ppo_aux[1], ppo_aux[2], gc_loss, gc_metrics))

                    minibatch_size = batch_size // config["num_minibatches"]
                    mb_idxs = jnp.arange(batch_size).reshape(config["num_minibatches"], minibatch_size)
                    minibatches = jax.tree_util.tree_map(lambda x: x[mb_idxs], batch)
                    mb_advantages = advantages_flat[mb_idxs]
                    mb_returns = returns_flat[mb_idxs]

                    (train_state, gc_train_state), losses = jax.lax.scan(
                        _update_minibatch, (train_state, gc_train_state), (minibatches, mb_advantages, mb_returns)
                    )
                    return (train_state, gc_train_state, rng), losses

                (train_state, gc_train_state, rng), losses = jax.lax.scan(
                    _update_epoch, (train_state, gc_train_state, rng), None, config["update_epochs"]
                )

                metrics = {
                    "total_loss": losses[0].mean(),
                    "pg_loss": losses[1][0].mean(),
                    "value_loss": losses[1][1].mean(),
                    "entropy": losses[1][2].mean(),
                    "gc_crl_loss": losses[1][3].mean(),
                    "gc_categorical_accuracy": losses[1][4]["gc_categorical_accuracy"].mean(),
                    "gc_logits_pos": losses[1][4]["gc_logits_pos"].mean(),
                    "gc_logits_neg": losses[1][4]["gc_logits_neg"].mean(),
                    "gc_infonce_loss": losses[1][4]["gc_infonce_loss"].mean(),
                    "gc_logsumexp_reg": losses[1][4]["gc_logsumexp_reg"].mean(),
                    "mean_reward": traj_batch.reward.mean(),
                }
                return (train_state, gc_train_state, env_state, last_obs, rng), metrics

            else:
                # Original path: baseline or contrastive
                def _update_epoch(carry, unused):
                    train_state, rng = carry
                    rng, perm_rng, contrastive_rng = jax.random.split(rng, 3)

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

                            if config["contrastive_coef"] > 0:
                                emb_reshaped = embedding.reshape(config["num_minibatches"], -1, config["embedding_dim"])
                                c_loss = contrastive_loss_fn(emb_reshaped, contrastive_rng)
                            else:
                                c_loss = 0.0

                            total_loss = (
                                pg_loss + config["vf_coef"] * value_loss
                                - config["ent_coef"] * entropy + config["contrastive_coef"] * c_loss
                            )
                            return total_loss, (pg_loss, value_loss, entropy, c_loss)

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
                    "contrastive_loss": losses[1][3].mean() if config["contrastive_coef"] > 0 else 0.0,
                    "mean_reward": traj_batch.reward.mean(),
                }
                return (train_state, env_state, last_obs, rng), metrics

        # Run training
        rng, train_rng = jax.random.split(rng)
        if use_gc_crl:
            runner_state = (train_state, gc_train_state, env_state, obsv, train_rng)
        else:
            runner_state = (train_state, env_state, obsv, train_rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["num_updates"])

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def run_experiment(config, seed=0):
    """Run a single experiment."""
    rng = jax.random.PRNGKey(seed)

    print(f"Creating training function...", flush=True)
    train_fn = make_train(config)
    print(f"JIT compiling (this may take a while on CPU)...", flush=True)
    train_jit = jax.jit(train_fn)

    print(f"Starting training with config: contrastive_coef={config['contrastive_coef']}", flush=True)
    start_time = time.time()

    out = train_jit(rng)

    # Block until computation is done
    print("Waiting for computation to complete...", flush=True)
    jax.block_until_ready(out)

    elapsed = time.time() - start_time

    # Extract final metrics
    final_metrics = jax.tree_util.tree_map(lambda x: float(x[-1]), out["metrics"])
    mean_metrics = jax.tree_util.tree_map(lambda x: float(x.mean()), out["metrics"])

    print(f"Training completed in {elapsed:.2f}s", flush=True)
    print(f"Final reward: {final_metrics['mean_reward']:.4f}", flush=True)
    print(f"Mean reward: {mean_metrics['mean_reward']:.4f}", flush=True)

    return {
        "final_metrics": final_metrics,
        "mean_metrics": mean_metrics,
        "elapsed_time": elapsed,
        "all_metrics": out["metrics"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "contrastive", "gc_crl"])
    parser.add_argument("--contrastive", action="store_true", help="Enable contrastive learning (legacy flag)")
    parser.add_argument("--contrastive_coef", type=float, default=0.001)
    # GC-CRL hyperparameters
    parser.add_argument("--gc_hidden_dim", type=int, default=1024, help="GC-CRL encoder hidden dim")
    parser.add_argument("--gc_embed_dim", type=int, default=64, help="GC-CRL embedding dim")
    parser.add_argument("--gc_contrastive_coef", type=float, default=0.1, help="GC-CRL loss coefficient")
    parser.add_argument("--gc_logsumexp_coef", type=float, default=0.1, help="GC-CRL logsumexp regularization")
    parser.add_argument("--gc_discount", type=float, default=0.99, help="GC-CRL geometric goal sampling discount")
    parser.add_argument("--gc_lr", type=float, default=3e-4, help="GC-CRL encoder learning rate")
    args = parser.parse_args()

    # Backward compat: --contrastive flag sets method
    if args.contrastive and args.method == "baseline":
        args.method = "contrastive"

    # Base config
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
        "method": args.method,
        "contrastive_coef": args.contrastive_coef if args.method == "contrastive" else 0.0,
        "contrastive_temperature": 0.1,
        "contrastive_gamma": 0.977,
        # GC-CRL config
        "gc_hidden_dim": args.gc_hidden_dim,
        "gc_embed_dim": args.gc_embed_dim,
        "gc_contrastive_coef": args.gc_contrastive_coef,
        "gc_logsumexp_coef": args.gc_logsumexp_coef,
        "gc_discount": args.gc_discount,
        "gc_lr": args.gc_lr,
    }

    # Calculate number of updates
    batch_size = config["num_envs"] * config["num_steps"]
    config["num_updates"] = args.total_timesteps // batch_size

    method_label = {"baseline": "Baseline", "contrastive": "+ Contrastive", "gc_crl": "+ GC-CRL"}[args.method]
    print("=" * 60, flush=True)
    print(f"Craftax PPO {method_label}", flush=True)
    print("=" * 60, flush=True)
    print(f"Total timesteps: {args.total_timesteps:,}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Num updates: {config['num_updates']}", flush=True)
    print(f"Method: {args.method}", flush=True)
    if args.method == "contrastive":
        print(f"Contrastive coef: {config['contrastive_coef']}", flush=True)
    elif args.method == "gc_crl":
        print(f"GC-CRL coef: {config['gc_contrastive_coef']}", flush=True)
    print("=" * 60, flush=True)

    # Run multiple seeds
    all_results = []
    for seed in range(args.num_seeds):
        print(f"\n--- Seed {seed} ---", flush=True)
        result = run_experiment(config, seed=seed)
        all_results.append(result)

    # Aggregate results
    mean_rewards = [r["mean_metrics"]["mean_reward"] for r in all_results]
    final_rewards = [r["final_metrics"]["mean_reward"] for r in all_results]

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Mean reward across seeds: {np.mean(mean_rewards):.4f} +/- {np.std(mean_rewards):.4f}", flush=True)
    print(f"Final reward across seeds: {np.mean(final_rewards):.4f} +/- {np.std(final_rewards):.4f}", flush=True)

    return all_results


if __name__ == "__main__":
    main()
