import argparse
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import wandb
import gymnasium as gym
from tqdm import tqdm
import torchvision

from .models.actor_critic import Actor, Critic
from .models.rssm import RSSM
from .models.vision import Decoder, Encoder
from .models.heads import RewardPredictor, DiscountPredictor
from .models.behavior import ImagBehavior
from .replay.replay_buffer import ReplayBuffer
from .envs.atari import Atari
from .utils.tools import lambda_return
from .utils.image_saver import save_reconstruction_predictions
import torchvision

import ale_py

def evaluate_agent(encoder, rssm, actor, eval_env, config, device):
    eval_episodes = config['training']['eval_eps']
    total_rewards = []
    episode_lengths = []
    action_dim = eval_env.action_space.n

    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        (prev_h, prev_z) = rssm.initial_state(1)

        while not done:
            with torch.no_grad():
                latent_state = torch.cat([prev_h, prev_z], dim=-1)
                action_dist = actor(latent_state.detach())
                action = action_dist.probs.argmax(dim=-1)
                action_np = action.cpu().numpy()[0]

            next_obs, reward, terminated, truncated, _ = eval_env.step(action_np)
            done = terminated or truncated
            obs = next_obs
            total_reward += reward
            episode_length += 1

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=device).unsqueeze(0).float() / 255.0 - 0.5
                obs_embed = encoder(obs_tensor)
                action_one_hot = F.one_hot(action, num_classes=action_dim).float()
                prev_h, prev_z, _, _ = rssm(obs_embed, action_one_hot, prev_h, prev_z)
        
        total_rewards.append(total_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_length = np.mean(episode_lengths)

    return mean_reward, std_reward, mean_length


def main(config):
    # comment this out if you don't want to use wandb
    if config['logging']['wandb']:
        wandb.init(project="dreamerv2-pytorch-refactored", config=config)

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    device_str = config['training'].get('device', 'cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"Using device: {device}")

    env = Atari(
        name=config['env']['name'],
        action_repeat=config['env']['action_repeat'],
        grayscale=config['env']['grayscale']
    )
    eval_env = Atari(
        name=config['env']['name'],
        action_repeat=config['env']['action_repeat'],
        grayscale=config['env']['grayscale']
    )
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    replay_buffer = ReplayBuffer(
        capacity=config['replay']['capacity'],
        sequence_length=config['replay']['sequence_length'],
        batch_size=config['training']['batch_size'],
        observation_shape=obs_shape,
        action_dim=1,  
        device=device,
        oversample_ends=config['replay'].get('oversample_ends', False)
    )

    stoch_size = config['model']['rssm']['category_size'] * config['model']['rssm']['class_size']
    deter_size = config['model']['rssm']['deter_size']
    embed_dim = config['model']['embed_dim'] 

    encoder = Encoder(embed_dim=embed_dim, in_channels=obs_shape[0]).to(device)
    rssm = RSSM(action_dim=action_dim, embed_dim=embed_dim, device=device, **config['model']['rssm']).to(device)
    decoder = Decoder(stoch_size=stoch_size, deter_size=deter_size, out_channels=obs_shape[0]).to(device)
    reward_predictor = RewardPredictor(stoch_size=stoch_size, deter_size=deter_size, hidden_size=config['model']['actor_critic']['hidden_size']).to(device)
    discount_predictor = DiscountPredictor(stoch_size=stoch_size, deter_size=deter_size, hidden_size=config['model']['actor_critic']['hidden_size']).to(device)
    actor = Actor(stoch_size=stoch_size, deter_size=deter_size, action_dim=action_dim, **config['model']['actor_critic']).to(device)
    critic = Critic(stoch_size=stoch_size, deter_size=deter_size, **config['model']['actor_critic']).to(device)

    world_model_params = list(encoder.parameters()) + list(rssm.parameters()) + list(decoder.parameters()) + list(reward_predictor.parameters()) + list(discount_predictor.parameters())

    behavior = ImagBehavior(config, rssm, actor, critic, device)

    world_model_optimizer = torch.optim.Adam(world_model_params, lr=float(config['training']['world_model_lr']), weight_decay=float(config['training']['weight_decay']))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(config['training']['actor_lr']), weight_decay=float(config['training']['weight_decay']))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(config['training']['critic_lr']), weight_decay=float(config['training']['weight_decay']))
    scaler = torch.amp.GradScaler(enabled=config['logging']['amp'])

    obs, _ = env.reset()
    done = False
    total_reward = 0
    cumulative_reward = 0
    episode_count = 0
    (prev_h, prev_z) = rssm.initial_state(1)
    episode_actions = []

    for frame in tqdm(range(1, config['training']['total_env_frames'] + 1), desc="Training Progress"):
        with torch.no_grad():
            latent_state = torch.cat([prev_h, prev_z], dim=-1)
            action = actor(latent_state.detach()).sample()
            action_np = action.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        episode_actions.append(action_np)
        replay_buffer.add(obs, action_np, reward, done)
        obs = next_obs
        total_reward += reward
        cumulative_reward += reward

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0).float() / 255.0 - 0.5
            obs_embed = encoder(obs_tensor)
            action_one_hot = F.one_hot(action, num_classes=action_dim).float()
            prev_h, prev_z, _, _ = rssm(obs_embed, action_one_hot, prev_h, prev_z)

        if done:
            episode_count += 1
            if config['logging']['wandb']:
                wandb.log({
                    'episode_reward': total_reward, 
                    'cumulative_reward': cumulative_reward,
                    'episode': episode_count, 
                    'frame': frame,
                    'action_histogram': wandb.Histogram(episode_actions)
                })
            episode_actions = []
            obs, _ = env.reset()
            done = False
            total_reward = 0
            (prev_h, prev_z) = rssm.initial_state(1)

        # --- Train World Model and Actor-Critic ---
        if frame > config['training']['burn_in_frames'] and frame % config['training']['world_model_update_interval'] == 0:
            batch = replay_buffer.sample()
            if batch is not None:
                world_model_optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=config['logging']['amp']):
                    obs_batch = batch['observations']
                    action_batch = F.one_hot(batch['actions'].long().squeeze(-1), num_classes=action_dim).float()
                    reward_batch = batch['rewards'].unsqueeze(-1)
                    if config['env'].get('clip_rewards') == 'tanh':
                        reward_batch = torch.tanh(reward_batch)
                    done_batch = batch['dones'].unsqueeze(-1)

                    embedded_obs = encoder(obs_batch.view(-1, *obs_shape)).view(
                        config['training']['batch_size'], config['replay']['sequence_length'], -1
                    )
                    if config['logging']['wandb']:
                        wandb.log({'debug/obs_embed_norm': torch.linalg.norm(embedded_obs).item()})

                    (h, z) = rssm.initial_state(config['training']['batch_size'])
                    h_states, z_states, kl_posts, kl_priors = [], [], [], []

                    for t in range(config['replay']['sequence_length']):
                        h, z, kl_post, kl_prior = rssm(embedded_obs[:, t], action_batch[:, t], h, z)
                        h_states.append(h)
                        z_states.append(z)
                        kl_posts.append(kl_post)
                        kl_priors.append(kl_prior)

                    h_states = torch.stack(h_states, dim=1)
                    z_states = torch.stack(z_states, dim=1)

                    kl_post = torch.stack(kl_posts).mean()
                    kl_prior = torch.stack(kl_priors).mean()
                    kl_balance = config['model']['rssm']['kl_balancing_alpha']
                    
                    kl_loss_post = kl_post
                    kl_loss_prior = kl_prior

                    # Free nats 
                    kl_free_nats = torch.tensor(config['model']['rssm'].get('kl_free', 0.0), device=device)
                    kl_loss_post = torch.max(kl_free_nats, kl_loss_post)
                    kl_loss_prior = torch.max(kl_free_nats, kl_loss_prior)
                    
                    kl_loss = (1 - kl_balance) * kl_loss_post + kl_balance * kl_loss_prior
                    
                    kl_loss *= config['model']['rssm']['kl_beta']
                    
                    latent_states = torch.cat([h_states, z_states], dim=-1)
                    
                    recon_dist = decoder(latent_states.view(-1, latent_states.shape[-1]))

                    recon_loss = -recon_dist.log_prob(obs_batch.view(-1, *obs_shape)).mean()

                    # Save reconstruction images 
                    save_after = config['logging'].get('save_recon_after_frames', 0)
                    stop_after = config['logging'].get('stop_save_recon_after_frames', np.inf)
                    save_interval = 50000

                    if config['logging'].get('save_recon_images', False) and \
                       frame >= save_after and \
                       frame <= stop_after and \
                       (frame - save_after) % save_interval == 0:
                        save_reconstruction_predictions(
                            recon_dist.mean.detach()[:16],
                            config['logging']['recon_image_dir'],
                            frame,
                            0
                        ) 
                    
                    reward_dist = reward_predictor(latent_states)
                    reward_loss = -reward_dist.log_prob(reward_batch).mean()
                    
                    discount_dist = discount_predictor(latent_states)
                    discount_target = (1.0 - done_batch.float())
                    discount_loss = -discount_dist.log_prob(discount_target).mean()

                    world_model_loss = recon_loss * config['model']['recon_scale'] + reward_loss + (discount_loss * config['model']['discount_scale']) + kl_loss

                scaler.scale(world_model_loss).backward()
                scaler.unscale_(world_model_optimizer)

                if config['logging']['wandb']:
                    encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), config['training']['model_grad_clip'])
                    rssm_grad_norm = torch.nn.utils.clip_grad_norm_(rssm.parameters(), config['training']['model_grad_clip'])
                    wandb.log({
                        'debug/encoder_grad_norm': encoder_grad_norm.item(),
                        'debug/rssm_grad_norm': rssm_grad_norm.item(),
                    })

                total_norm = torch.nn.utils.clip_grad_norm_(world_model_params, config['training']['model_grad_clip'])
                scaler.step(world_model_optimizer)
                scaler.update()

                
                # --- Actor-Critic Update ---
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=config['logging']['amp']):
                    imag_h_states = h_states.detach()
                    imag_z_states = z_states.detach()
                    # if config['model'].get('pred_discount', False):
                    #     imag_h_states = imag_h_states[:, :-1]
                    #     imag_z_states = imag_z_states[:, :-1]

                    actor_loss, critic_loss, entropy = behavior.train_step(imag_h_states, imag_z_states, reward_predictor, discount_predictor, frame)

                scaler.scale(actor_loss).backward()
                scaler.scale(critic_loss).backward()
                scaler.unscale_(actor_optimizer)
                scaler.unscale_(critic_optimizer)
                torch.nn.utils.clip_grad_norm_(actor.parameters(), config['training']['actor_grad_clip'])
                torch.nn.utils.clip_grad_norm_(critic.parameters(), config['training']['value_grad_clip'])
                scaler.step(actor_optimizer)
                scaler.step(critic_optimizer)
                scaler.update()

                if config['logging']['wandb']:
                    wandb.log({
                        'world_model_loss': world_model_loss.item(),
                        'recon_loss': recon_loss.item(),
                        'kl_loss': kl_loss.item(),
                        'reward_loss': reward_loss.item(),
                        'discount_loss': discount_loss.item(),
                        'kl_post': kl_post.item(),
                        'kl_prior': kl_prior.item(),
                        'grad_norm': total_norm.item(),
                        'actor_loss': actor_loss.item(),
                        'critic_loss': critic_loss.item(),
                        'entropy': entropy,
                        'frame': frame
                    })

        # --- Evaluation ---
        if frame % config['training']['eval_every'] == 0 and frame > 0:
            print(f"\n--- Starting evaluation at frame {frame} ---")
            encoder.eval()
            rssm.eval()
            actor.eval()
            critic.eval()
            
            mean_reward, std_reward, mean_length = evaluate_agent(encoder, rssm, actor, eval_env, config, device)
            
            if config['logging']['wandb']:
                wandb.log({
                    'eval_mean_reward': mean_reward,
                    'eval_std_reward': std_reward,
                    'eval_mean_length': mean_length,
                    'frame': frame
                })
            print(f"Evaluation complete: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}, Mean Length: {mean_length:.2f}")
            print("--- Resuming training ---")
            
            encoder.train()
            rssm.train()
            actor.train()
            critic.train()

        # --- Checkpointing ---
        if (frame % config['training']['checkpoint_interval_steps'] == 0 and frame > 0) or frame == 100000:
            checkpoint_dir = pathlib.Path(config['logging']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"model_{frame}.pth"
            torch.save({
                'encoder': encoder.state_dict(),
                'rssm': rssm.state_dict(),
                'decoder': decoder.state_dict(),
                'reward_predictor': reward_predictor.state_dict(),
                'discount_predictor': discount_predictor.state_dict(),
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'world_model_optimizer': world_model_optimizer.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'critic_optimizer': critic_optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'frame': frame,
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pong.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['seed'] = config.get('seed', 42)
    config['env']['name'] = config['env'].get('name', 'ALE/Pong-v5')
    main(config)