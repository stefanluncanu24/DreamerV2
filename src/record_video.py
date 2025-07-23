import argparse
import os
import random
from pathlib import Path
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import gymnasium as gym
from tqdm import tqdm
import ale_py

if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.models.actor_critic import Actor
    from src.models.rssm import RSSM
    from src.models.vision import Encoder
    from src.envs.atari import Atari
else:
    from .models.actor_critic import Actor
    from .models.rssm import RSSM
    from .models.vision import Encoder
    from .envs.atari import Atari

def record_and_find_longest(config, checkpoint_path, output_dir, num_episodes):
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device_str = config['training'].get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"Using device: {device}")

    model_name = Path(checkpoint_path).parent.name
    checkpoint_name = Path(checkpoint_path).stem
    temp_video_dir = Path(output_dir) / f"temp_{model_name}_{checkpoint_name}_{random.randint(1000, 9999)}"
    temp_video_dir.mkdir(parents=True, exist_ok=True)

    # Create the base environment with render_mode for video recording
    base_env = gym.make(config['env']['name'], render_mode='rgb_array', frameskip=1)
    
    # Apply the Atari preprocessing wrapper from gym
    processed_env = gym.wrappers.AtariPreprocessing(
        base_env,
        noop_max=config['env'].get('noops', 30),
        frame_skip=config['env']['action_repeat'],
        screen_size=config['env'].get('size', [64, 64])[0],
        terminal_on_life_loss=config['env'].get('life_done', False),
        grayscale_obs=config['env'].get('grayscale', True)
    )

    # Wrap for video recording
    env = gym.wrappers.RecordVideo(processed_env, video_folder=str(temp_video_dir), episode_trigger=lambda x: True)

    obs_shape = (1, 64, 64) if config['env'].get('grayscale', True) else (3, 64, 64)
    action_dim = env.action_space.n
    stoch_size = config['model']['rssm']['category_size'] * config['model']['rssm']['class_size']
    deter_size = config['model']['rssm']['deter_size']
    embed_dim = config['model']['embed_dim']

    encoder = Encoder(embed_dim=embed_dim, in_channels=obs_shape[0]).to(device)
    rssm = RSSM(action_dim=action_dim, embed_dim=embed_dim, device=device, **config['model']['rssm']).to(device)
    actor = Actor(stoch_size=stoch_size, deter_size=deter_size, action_dim=action_dim, **config['model']['actor_critic']).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    rssm.load_state_dict(checkpoint['rssm'])
    actor.load_state_dict(checkpoint['actor'])

    encoder.eval()
    rssm.eval()
    actor.eval()

    episode_lengths = []
    print(f"Recording {num_episodes} episodes to find the longest one...")

    for i in tqdm(range(num_episodes), desc="Recording Episodes"):
        obs, _ = env.reset()
        done = False
        episode_length = 0
        (prev_h, prev_z) = rssm.initial_state(1)

        while not done:
            with torch.no_grad():
                latent_state = torch.cat([prev_h, prev_z], dim=-1)
                action = actor(latent_state.detach()).sample()
                action_np = action.cpu().numpy()[0]

            next_obs, _, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            # Ensure observation is in the correct shape for the model
            if len(next_obs.shape) == 2: # (H, W) -> (1, H, W)
                next_obs = np.expand_dims(next_obs, axis=0)

            obs = next_obs
            episode_length += 1

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=device).unsqueeze(0).float() / 255.0 - 0.5
                obs_embed = encoder(obs_tensor)
                action_one_hot = F.one_hot(action, num_classes=action_dim).float()
                # Corrected the call to rssm to match its forward signature
                prev_h, prev_z, _, _ = rssm(obs_embed, action_one_hot, prev_h, prev_z)
        
        episode_lengths.append(episode_length)

    env.close()

    if not episode_lengths:
        print("No episodes were recorded.")
        shutil.rmtree(temp_video_dir)
        return

    longest_episode_index = np.argmax(episode_lengths)
    longest_length = episode_lengths[longest_episode_index]
    print(f"\nLongest episode: #{longest_episode_index} with length {longest_length}")

    video_files = sorted(list(temp_video_dir.glob("*.mp4")))

    if longest_episode_index < len(video_files):
        longest_video_path = video_files[longest_episode_index]
        
        final_output_dir = Path(output_dir) / model_name / checkpoint_name
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_video_path = final_output_dir / f"longest_of_{num_episodes}_length_{longest_length}.mp4"

        shutil.move(str(longest_video_path), str(final_video_path))
        print(f"Saved longest episode video to: {final_video_path}")
    else:
        print(f"Error: Could not find the video file for the longest episode (index {longest_episode_index}).")

    shutil.rmtree(temp_video_dir)
    print("Cleaned up temporary video files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file (e.g., configs/breakout.yaml).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file (.pth).")
    parser.add_argument("--output-dir", type=str, default="recordings", help="Directory to save the recorded videos.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to record to find the longest one.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    record_and_find_longest(config, args.checkpoint, args.output_dir, args.num_episodes)
