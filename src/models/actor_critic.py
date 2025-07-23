import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import Categorical

def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Critic(nn.Module):
    def __init__(self, stoch_size, deter_size, hidden_size=400, **kwargs):
        super().__init__()
        self.latent_dim = stoch_size + deter_size
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2)
        )
        self.apply(orthogonal_init)

    def forward(self, latent_state):
        mean, std = self.model(latent_state).chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        return dist.Independent(dist.Normal(mean, std), 1)


class Actor(nn.Module):
    def __init__(self, stoch_size, deter_size, action_dim, hidden_size=400, **kwargs):
        super().__init__()
        self.latent_dim = stoch_size + deter_size
        self.action_dim = action_dim
        self.entropy_coef = kwargs.get('entropy_coef', 0.003) 
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.action_dim)
        )
        self.apply(orthogonal_init)

    def forward(self, latent_state):
        logits = self.model(latent_state)
        return Categorical(logits=logits)

    def get_action(self, latent_state, deterministic=False):
        dist = self.forward(latent_state)
        if deterministic:
            return dist.probs.argmax(dim=-1)
        return dist.sample()
