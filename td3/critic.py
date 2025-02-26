import torch 
from torch import nn
import torch.nn.functional as F



class TD3Critic(nn.Module):
    """
    TD3 Critic with twin Q-networks to reduce overestimation bias.
    """

    def __init__(self, model_kwargs, state_dim, action_dim):
        super(TD3Critic, self).__init__()

        hidden_size = model_kwargs.get('hidden_size', 256)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)

        # Q1 Network
        layers1 = [nn.Linear(state_dim + action_dim, hidden_size), nn.ELU()]
        for _ in range(num_hidden_layers - 1):
            layers1.extend([nn.Linear(hidden_size, hidden_size), nn.ELU()])
        layers1.append(nn.Linear(hidden_size, 1))
        self.net1 = nn.Sequential(*layers1)  

        # Q2 Network
        layers2 = [nn.Linear(state_dim + action_dim, hidden_size), nn.ELU()]
        for _ in range(num_hidden_layers - 1):
            layers2.extend([nn.Linear(hidden_size, hidden_size), nn.ELU()])
        layers2.append(nn.Linear(hidden_size, 1))
        self.net2 = nn.Sequential(*layers2)  

    def forward(self, state, action, return_all=True):
        x = torch.cat([state, action], dim=1)
        q1 = self.net1(x)
        q2 = self.net2(x)
        # Return both Q-values for critic loss computation 
        # Return only q1 for actor update 
        return (q1, q2) if return_all else q1 
