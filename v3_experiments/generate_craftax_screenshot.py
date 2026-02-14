"""
Generate a Craftax screenshot by rendering a frame from a short episode.

Uses render_craftax_pixels() from the Craftax library.
"""
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG


def generate_craftax_screenshot(output_dir, seed=0, target_step=100):
    """Render a Craftax frame at a given step."""
    os.makedirs(output_dir, exist_ok=True)

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)

    # Run random actions to get an interesting game state
    for step in range(target_step):
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        action = jax.random.randint(action_rng, (), 0, env.action_space(env_params).n)
        obs, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)
        if done:
            # Reset and keep going
            rng, reset_rng = jax.random.split(rng)
            obs, env_state = env.reset(reset_rng, env_params)

    # Render the current state
    pixels = render_craftax_pixels(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG)
    frame = np.array(pixels).astype(np.uint8)

    # Save as image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frame)
    ax.set_title(f"Craftax Environment (Step {target_step})", fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/craftax_screenshot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Craftax screenshot to {output_dir}/craftax_screenshot.png")

    # Also save raw frame without matplotlib decoration
    from PIL import Image
    img = Image.fromarray(frame)
    img.save(f"{output_dir}/craftax_screenshot_raw.png")
    print(f"Saved raw Craftax frame to {output_dir}/craftax_screenshot_raw.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/home/ubuntu/metta/v3_experiments/figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=100)
    args = parser.parse_args()
    generate_craftax_screenshot(args.output_dir, seed=args.seed, target_step=args.step)
