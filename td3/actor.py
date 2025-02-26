from torch import nn
import torch
import torch.nn.functional as F

# Actor network for deterministic policy


class TD3Actor(nn.Module):
    """
    TD3 Actor network for deterministic policy.
    """

    def __init__(self, model_kwargs, state_dim, action_dim, max_action):
        super(TD3Actor, self).__init__()

        hidden_size = model_kwargs.get("hidden_size", 256)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)

        layers = [nn.Linear(state_dim, hidden_size), nn.ELU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ELU()])

        layers.append(nn.Linear(hidden_size, action_dim))  # Deterministic action output
        self.net = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * torch.tanh(self.net(state))  # Ensure action bounds

    def get_action(self, state, noise=0.1):
        action = self.forward(state)
        action += torch.randn_like(action) * noise  # Exploration noise
        return action.clamp(-self.max_action, self.max_action)
