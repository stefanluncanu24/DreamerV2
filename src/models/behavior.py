import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 

from .actor_critic import Actor, Critic
from ..utils.tools import lambda_return, schedule

class ImagBehavior(nn.Module):
    def __init__(self, config, rssm, actor, critic, device):
        super().__init__()
        self.config = config
        self.rssm = rssm
        self.actor = actor
        self.critic = critic
        self.device = device
        self.stoch_size = config['model']['rssm']['category_size'] * config['model']['rssm']['class_size']
        self.deter_size = config['model']['rssm']['deter_size']
        self.action_dim = actor.action_dim
        self.critic_target = copy.deepcopy(critic)
        self.updates = 0

    def train_step(self, h_states, z_states, reward_predictor, discount_predictor, step):
        self._update_critic_target()

        imag_h_states = h_states.detach().reshape(-1, self.deter_size)
        imag_z_states = z_states.detach().reshape(-1, self.stoch_size)

        imagined_h, imagined_z, imagined_actions = self._imagine_trajectories(
            imag_h_states, imag_z_states
        )

        imagined_latent_states = torch.cat([imagined_h, imagined_z], dim=-1)

        imagined_rewards = reward_predictor(imagined_latent_states).mean
        imagined_discounts = discount_predictor(imagined_latent_states).mean
        imagined_values = self.critic_target(imagined_latent_states).mean.detach()

        lambda_returns = lambda_return(
            imagined_rewards.squeeze(-1),
            imagined_values.squeeze(-1),
            imagined_discounts.squeeze(-1),
            bootstrap=imagined_values[-1].squeeze(-1),
            lambda_=self.config['training']['lambda_return']
        )

        imag_gradient_strat = self.config['training'].get('imag_gradient', 'dynamics')

        actor_latent_states = imagined_latent_states[:-1].detach()
        policy = self.actor(actor_latent_states)

        if imag_gradient_strat == 'dynamics':
            objective = lambda_returns[:-1]
        
        elif imag_gradient_strat == 'reinforce':
            baseline = self.critic(actor_latent_states).mean.squeeze(-1)
            advantage = (lambda_returns[:-1] - baseline).detach()
            objective = policy.log_prob(imagined_actions.detach()) * advantage
        
        elif imag_gradient_strat == 'both':
            baseline = self.critic(actor_latent_states).mean.squeeze(-1)
            advantage = (lambda_returns[:-1] - baseline).detach()
            reinforce_objective = policy.log_prob(imagined_actions.detach()) * advantage
            
            dynamics_objective = lambda_returns[:-1]
            
            mix_ratio = schedule(self.config['training']['imag_gradient_mix'], step)
            
            objective = (1 - mix_ratio) * dynamics_objective + mix_ratio * reinforce_objective
        else:
            raise NotImplementedError(f"Unknown imag_gradient: {imag_gradient_strat}")

        entropy_coeff = schedule(self.config['model']['actor_critic']['entropy_coeff'], step)
        objective += entropy_coeff * policy.entropy()

        actor_loss = -torch.mean(objective)

        critic_latent_states = imagined_latent_states[:-1].detach()
        critic_dist = self.critic(critic_latent_states)
        critic_loss = -critic_dist.log_prob(lambda_returns[:-1].detach().unsqueeze(-1)).mean()

        return actor_loss, critic_loss, policy.entropy().mean().item()

    def _imagine_trajectories(self, initial_h, initial_z):
        imagined_h, imagined_z, imagined_actions = [initial_h], [initial_z], []

        for _ in range(self.config['training']['imagination_horizon'] - 1):
            latent_state = torch.cat([imagined_h[-1], imagined_z[-1]], dim=-1)
            action_dist = self.actor(latent_state.detach())
            action = action_dist.sample()
            action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
            
            imagined_actions.append(action)
            
            h, z = self.rssm.imagine(action_one_hot, imagined_h[-1], imagined_z[-1])
            imagined_h.append(h)
            imagined_z.append(z)

        imagined_h = torch.stack(imagined_h, dim=0)
        imagined_z = torch.stack(imagined_z, dim=0)
        imagined_actions = torch.stack(imagined_actions, dim=0)

        return imagined_h, imagined_z, imagined_actions

    def _update_critic_target(self):
        if self.config['training'].get('slow_value_target', False):
            if self.updates % self.config['training']['slow_target_update'] == 0:
                mix = self.config['training'].get('slow_target_fraction', 1.0)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)
            self.updates += 1
