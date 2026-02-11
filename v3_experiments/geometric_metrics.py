"""
Geometric diagnostic metrics for MARL representation analysis.

Implements three metrics from the paper:
1. Effective Rank - entropy of normalized singular value spectrum
2. Expansion Ratio - variance ratio of late vs early trajectory embeddings
3. Value Representation Rank - effective rank of critic hidden layer

Both PyTorch and JAX implementations provided.
"""

import numpy as np

# ============================================================================
# NumPy implementations (work with both frameworks after .numpy() conversion)
# ============================================================================


def effective_rank(embeddings: np.ndarray) -> float:
    """Compute effective rank of an embedding matrix via SVD entropy.

    EffRank(Z) = exp(-sum(sigma_bar_i * log(sigma_bar_i)))
    where sigma_bar_i = sigma_i / sum(sigma_j)

    Args:
        embeddings: (batch_size, embedding_dim) array

    Returns:
        Effective rank (1.0 = fully collapsed, d = full rank)
    """
    if embeddings.shape[0] < 2:
        return float(embeddings.shape[1])

    # Center embeddings
    embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)

    # SVD
    try:
        _, sigmas, _ = np.linalg.svd(embeddings, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0

    # Normalize singular values
    sigma_sum = sigmas.sum()
    if sigma_sum < 1e-10:
        return 1.0

    sigma_bar = sigmas / sigma_sum
    # Filter zeros for log stability
    sigma_bar = sigma_bar[sigma_bar > 1e-10]

    # Shannon entropy -> effective rank
    entropy = -np.sum(sigma_bar * np.log(sigma_bar))
    return float(np.exp(entropy))


def expansion_ratio(embeddings: np.ndarray, early_frac: float = 0.2, late_frac: float = 0.8) -> float:
    """Compute trajectory expansion ratio.

    ExpansionRatio = Var(z_t for t > late_frac * T) / Var(z_t for t < early_frac * T)

    Args:
        embeddings: (num_steps, embedding_dim) array for a single trajectory
        early_frac: fraction of trajectory considered "early"
        late_frac: fraction of trajectory considered "late"

    Returns:
        Expansion ratio (>1 = expansion near goal, <1 = contraction)
    """
    T = embeddings.shape[0]
    if T < 5:
        return 1.0

    early_cutoff = max(1, int(early_frac * T))
    late_cutoff = max(early_cutoff + 1, int(late_frac * T))

    early_emb = embeddings[:early_cutoff]
    late_emb = embeddings[late_cutoff:]

    if len(early_emb) < 2 or len(late_emb) < 2:
        return 1.0

    early_var = np.var(early_emb)
    late_var = np.var(late_emb)

    if early_var < 1e-10:
        return float(late_var / 1e-10)

    return float(late_var / early_var)


def svd_spectrum(embeddings: np.ndarray, top_k: int = 64) -> np.ndarray:
    """Get top-k singular values of embedding matrix.

    Args:
        embeddings: (batch_size, embedding_dim) array
        top_k: number of singular values to return

    Returns:
        Array of top-k singular values (normalized)
    """
    embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)
    try:
        _, sigmas, _ = np.linalg.svd(embeddings, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(min(top_k, embeddings.shape[1]))

    sigmas = sigmas[:top_k]
    sigma_sum = sigmas.sum()
    if sigma_sum > 1e-10:
        sigmas = sigmas / sigma_sum
    return sigmas


def sigma_ratio(embeddings: np.ndarray, k: int = 10) -> float:
    """Compute sigma_1 / sigma_k ratio as collapse indicator.

    Args:
        embeddings: (batch_size, embedding_dim) array
        k: index for denominator singular value

    Returns:
        Ratio sigma_1/sigma_k (high = collapsed)
    """
    embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)
    try:
        _, sigmas, _ = np.linalg.svd(embeddings, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("inf")

    if len(sigmas) < k or sigmas[k - 1] < 1e-10:
        return float("inf")

    return float(sigmas[0] / sigmas[k - 1])


# ============================================================================
# PyTorch helpers (for SMAC and MettaGrid scripts)
# ============================================================================


def compute_geometric_metrics_torch(
    embeddings_buf,
    values_buf,
    dones_buf,
    critic_hidden=None,
    num_envs=1,
    n_agents=1,
):
    """Compute all geometric metrics from PyTorch training buffers.

    Args:
        embeddings_buf: (num_steps, batch_size, embedding_dim) tensor
        values_buf: (num_steps, batch_size) tensor
        dones_buf: (num_steps, batch_size) tensor
        critic_hidden: optional (batch_size, hidden_dim) tensor from critic's penultimate layer
        num_envs: number of parallel environments
        n_agents: agents per environment

    Returns:
        dict of metric name -> float
    """
    import torch

    metrics = {}

    # Convert to numpy for SVD computation
    emb_np = embeddings_buf.detach().cpu().numpy()
    num_steps, batch_size, embed_dim = emb_np.shape

    # 1. Effective rank of all embeddings (flatten steps and batch)
    all_emb = emb_np.reshape(-1, embed_dim)
    # Subsample if too large
    if all_emb.shape[0] > 4096:
        idx = np.random.choice(all_emb.shape[0], 4096, replace=False)
        all_emb = all_emb[idx]
    metrics["geom/effective_rank"] = effective_rank(all_emb)
    metrics["geom/effective_rank_normalized"] = metrics["geom/effective_rank"] / embed_dim

    # 2. SVD spectrum and sigma ratio
    metrics["geom/sigma1_sigma10_ratio"] = sigma_ratio(all_emb, k=10)
    spectrum = svd_spectrum(all_emb, top_k=min(20, embed_dim))
    for i, s in enumerate(spectrum[:5]):
        metrics[f"geom/sigma_{i+1}"] = float(s)

    # 3. Expansion ratio - compute per-environment trajectories
    dones_np = dones_buf.detach().cpu().numpy()
    agents_per_env = n_agents
    exp_ratios = []

    for env_idx in range(num_envs):
        for agent_idx in range(agents_per_env):
            flat_idx = env_idx * agents_per_env + agent_idx
            if flat_idx >= batch_size:
                break
            traj_emb = emb_np[:, flat_idx, :]  # (num_steps, embed_dim)
            er = expansion_ratio(traj_emb)
            if np.isfinite(er):
                exp_ratios.append(er)

    if exp_ratios:
        metrics["geom/expansion_ratio_mean"] = float(np.mean(exp_ratios))
        metrics["geom/expansion_ratio_std"] = float(np.std(exp_ratios))
        metrics["geom/expansion_ratio_gt1_frac"] = float(np.mean([1.0 if r > 1.0 else 0.0 for r in exp_ratios]))
    else:
        metrics["geom/expansion_ratio_mean"] = 1.0
        metrics["geom/expansion_ratio_std"] = 0.0
        metrics["geom/expansion_ratio_gt1_frac"] = 0.5

    # 4. Value representation rank (if critic hidden provided)
    if critic_hidden is not None:
        vh_np = critic_hidden.detach().cpu().numpy()
        if vh_np.shape[0] > 4096:
            idx = np.random.choice(vh_np.shape[0], 4096, replace=False)
            vh_np = vh_np[idx]
        metrics["geom/value_rank"] = effective_rank(vh_np)
        metrics["geom/value_rank_normalized"] = metrics["geom/value_rank"] / vh_np.shape[1]

    return metrics


# ============================================================================
# JAX/NumPy helpers (for Craftax script)
# ============================================================================


def compute_geometric_metrics_jax(
    embeddings,
    dones=None,
    num_envs=1,
):
    """Compute geometric metrics from JAX arrays.

    Args:
        embeddings: (num_steps, num_envs, embedding_dim) numpy array
        dones: optional (num_steps, num_envs) numpy array
        num_envs: number of environments

    Returns:
        dict of metric name -> float
    """
    # Ensure numpy
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    num_steps, n_envs, embed_dim = embeddings.shape
    metrics = {}

    # 1. Effective rank
    all_emb = embeddings.reshape(-1, embed_dim)
    if all_emb.shape[0] > 4096:
        idx = np.random.choice(all_emb.shape[0], 4096, replace=False)
        all_emb = all_emb[idx]
    metrics["geom/effective_rank"] = effective_rank(all_emb)
    metrics["geom/effective_rank_normalized"] = metrics["geom/effective_rank"] / embed_dim

    # 2. SVD spectrum
    metrics["geom/sigma1_sigma10_ratio"] = sigma_ratio(all_emb, k=min(10, embed_dim))
    spectrum = svd_spectrum(all_emb, top_k=min(20, embed_dim))
    for i, s in enumerate(spectrum[:5]):
        metrics[f"geom/sigma_{i+1}"] = float(s)

    # 3. Expansion ratio per environment
    exp_ratios = []
    for env_idx in range(n_envs):
        traj_emb = embeddings[:, env_idx, :]
        er = expansion_ratio(traj_emb)
        if np.isfinite(er):
            exp_ratios.append(er)

    if exp_ratios:
        metrics["geom/expansion_ratio_mean"] = float(np.mean(exp_ratios))
        metrics["geom/expansion_ratio_std"] = float(np.std(exp_ratios))
        metrics["geom/expansion_ratio_gt1_frac"] = float(np.mean([1.0 if r > 1.0 else 0.0 for r in exp_ratios]))
    else:
        metrics["geom/expansion_ratio_mean"] = 1.0
        metrics["geom/expansion_ratio_std"] = 0.0
        metrics["geom/expansion_ratio_gt1_frac"] = 0.5

    return metrics
