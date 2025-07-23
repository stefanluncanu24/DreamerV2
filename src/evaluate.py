import argparse
import os
import random
import multiprocessing
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import gymnasium as gym
from tqdm import tqdm

import ale_py

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.models.actor_critic import Actor, Critic
    from src.models.rssm import RSSM
    from src.models.vision import Encoder
    from src.envs.atari import Atari
else:
    from .models.actor_critic import Actor
    from .models.rssm import RSSM
    from .models.vision import Encoder
    from .envs.atari import Atari


def plot_cumulative_rewards(cumulative_rewards_per_episode, checkpoint_name, save_dir):
    plt.figure(figsize=(12, 8))

    if not cumulative_rewards_per_episode:
        print(f"No reward data to plot for {checkpoint_name}.")
        plt.close()
        return

    max_len = min(max(len(ep) for ep in cumulative_rewards_per_episode), 1000)
    padded_rewards = np.array([
        np.pad(ep[:max_len], (0, max_len - len(ep[:max_len])), 'edge')
        for ep in cumulative_rewards_per_episode
    ])

    if padded_rewards.size == 0:
        print(f"No reward data to plot for {checkpoint_name}.")
        plt.close()
        return

    mean_rewards = np.mean(padded_rewards, axis=0)
    std_rewards = np.std(padded_rewards, axis=0)

    timesteps = np.arange(max_len)

    plt.plot(timesteps, mean_rewards, label='Mean Cumulative Reward')
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label='Standard Deviation'
    )

    plt.xlabel("Time Steps (capped at 1000)")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Cumulative Reward for {checkpoint_name}")
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{checkpoint_name}_cumulative_reward.png"))
    plt.close()

def run_evaluation_for_checkpoint(args_tuple):
    config, checkpoint_path, total_steps, worker_id, output_dir = args_tuple
    torch.manual_seed(config['seed'] + worker_id)
    random.seed(config['seed'] + worker_id)
    np.random.seed(config['seed'] + worker_id)
    device_str = config['training'].get('device', 'cuda:3' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    env = Atari(
        name=config['env']['name'],
        action_repeat=config['env']['action_repeat']
    )

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    stoch_size = config['model']['rssm']['category_size'] * config['model']['rssm']['class_size']
    deter_size = config['model']['rssm']['deter_size']
    embed_dim = config['model']['embed_dim']

    encoder = Encoder(embed_dim=embed_dim, in_channels=obs_shape[0]).to(device)
    rssm = RSSM(action_dim=action_dim, embed_dim=embed_dim, device=device, **config['model']['rssm']).to(device)
    actor = Actor(stoch_size=stoch_size, deter_size=deter_size, action_dim=action_dim, **config['model']['actor_critic']).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    rssm.load_state_dict(checkpoint['rssm'])
    actor.load_state_dict(checkpoint['actor'])

    encoder.eval()
    rssm.eval()
    actor.eval()

    total_rewards_per_episode = []
    cumulative_rewards_per_episode = []
    steps_done = 0

    pbar = tqdm(total=total_steps, desc=f"Worker {worker_id} Evaluating {checkpoint_path.name}", leave=False)
    while steps_done < total_steps:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        cumulative_rewards = []
        (prev_h, prev_z) = rssm.initial_state(1)

        while not done and steps_done < total_steps:
            with torch.no_grad():
                latent_state = torch.cat([prev_h, prev_z], dim=-1)
                action = actor(latent_state.detach()).sample()
                action_np = action.cpu().numpy()[0]

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            obs = next_obs
            total_reward += reward
            cumulative_rewards.append(total_reward)
            steps_done += 1
            pbar.update(1)

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=device).unsqueeze(0).float() / 255.0 - 0.5
                obs_embed = encoder(obs_tensor)
                action_one_hot = F.one_hot(action, num_classes=action_dim).float()
                prev_h, prev_z, _, _ = rssm(obs_embed, action_one_hot, prev_h, prev_z)

        total_rewards_per_episode.append(total_reward)
        cumulative_rewards_per_episode.append(np.array(cumulative_rewards))
    
    pbar.close()
    env.close()

    checkpoint_name = checkpoint_path.stem
    plot_cumulative_rewards(cumulative_rewards_per_episode, checkpoint_name, output_dir)

    return checkpoint_name, np.mean(total_rewards_per_episode), np.std(total_rewards_per_episode)

def evaluate(config, checkpoint_dir, total_steps, num_workers):
    checkpoint_dir = Path(checkpoint_dir)
    eval_run_name = checkpoint_dir.name
    output_dir = Path("evaluation_plots") / eval_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = sorted([f for f in checkpoint_dir.glob('*.pth')])
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    num_workers = min(num_workers, len(checkpoint_files))

    pool_args = [
        (config, cp_path, total_steps, i, output_dir)
        for i, cp_path in enumerate(checkpoint_files)
    ]

    print(f"Starting evaluation for {len(checkpoint_files)} checkpoints with {num_workers} workers.")
    
    results = []
    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(run_evaluation_for_checkpoint, pool_args), total=len(pool_args), desc=f"Evaluating {eval_run_name}"))
    else: 
        for args in tqdm(pool_args, desc=f"Evaluating {eval_run_name}"):
            results.append(run_evaluation_for_checkpoint(args))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- Comparative Analysis for {eval_run_name} ---")
    print(f"{ 'Checkpoint':<40} {'Mean Reward':<20} {'Std Reward':<20}")
    print("-" * 80)
    for name, mean, std in results:
        print(f"{name:<40} {mean:<20.2f} {std:<20.2f}")
    print("-" * 80)

    names = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]

    plt.figure(figsize=(max(12, len(names) * 0.5), 8))
    plt.bar(names, means, yerr=stds, capsize=5, color='skyblue', ecolor='gray')
    plt.xlabel("Checkpoint")
    plt.ylabel("Mean Total Reward per Episode")
    plt.title(f"Comparative Analysis for {eval_run_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "comparative_analysis.png")
    plt.close()
    print(f"Saved comparative analysis plot to {output_dir / 'comparative_analysis.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to the directory containing model checkpoints.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total environment steps to evaluate for each checkpoint.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for evaluation. Note: A high number of workers on a single GPU can lead to memory issues.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    evaluate(config, args.checkpoint_dir, args.steps, args.workers)
