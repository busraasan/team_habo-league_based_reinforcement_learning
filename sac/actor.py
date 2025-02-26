from torch import nn
import torch
import torch.nn.functional as F

#! So since it is a continous action space, we have to use a gaussian policy or a deterministic policy.
#! In the gaussian policy, we have to output the mean and the log_std of the action. 
#! In the deterministic policy, we have to output the action directly.
#! But to use entropy regularization, we have to use the gaussian policy. 


class SacActor(nn.Module):
    """
    SAC Actor network for continuous action spaces.
    Supports both Gaussian (stochastic) and Deterministic policies.
    """

    def __init__(self, model_kwargs, state_dim, action_dim, max_action):
        """
        Initializes the SAC Actor.

        Args:
            model_kwargs (dict): Contains hyperparameters (hidden_size, num_hidden_layers, policy_type).
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            max_action (float): The max possible action magnitude.
        """
        super(SacActor, self).__init__()

        hidden_size = model_kwargs.get("hidden_size", 256)
        num_hidden_layers = model_kwargs.get("num_hidden_layers", 2)
        self.policy_type = model_kwargs.get("policy_type", "Gaussian")

        layers = [nn.Linear(state_dim, hidden_size), nn.ELU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ELU()])

        if self.policy_type == "Gaussian":
            layers.append(nn.Linear(hidden_size, 2 * action_dim))  # Output mean and log_std
        elif self.policy_type == "Deterministic":
            layers.append(nn.Linear(hidden_size, action_dim))  # Output action directly
        else:
            raise ValueError("Policy type not recognized. Please use 'Gaussian' or 'Deterministic'.")

        self.net = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        """
        Forward pass of the actor network.
        Returns mean and log_std if Gaussian, else returns action directly.

        Args:
            state (torch.Tensor): State tensor (Batch_size x State_dim).

        Returns:
            torch.Tensor: Mean and log_std for Gaussian policy, or action for Deterministic policy.
        """
        output = self.net(state)

        if self.policy_type == "Gaussian":
            mean, log_std = torch.chunk(output, 2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)  # Stabilize std range
            return mean, log_std
        elif self.policy_type == "Deterministic":
            return self.max_action * torch.tanh(output)

    def sample_action(self, state):
        """
        Samples an action given the state using the reparameterization trick.

        Args:
            state (torch.Tensor): State tensor.

        Returns:
            action (torch.Tensor): Sampled action tensor.
            log_prob (torch.Tensor): Log probability of the sampled action which will be used in the loss.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()  # Convert log_std to std

        # Gaussian Sampling (Reparameterization Trick)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Sample with gradients
        y_t = torch.tanh(x_t)  # Apply tanh squashing
        action = y_t * self.max_action

        # Compute log_prob with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)  # Tanh correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # Sum across action dimensions

        return action, log_prob

    def get_action(self, state, deterministic=False):
        """
        Returns an action given a state.

        Args:
            state (torch.Tensor): State tensor.
            deterministic (bool): Whether to use deterministic mode.

        Returns:
            torch.Tensor: The selected action.
        """

        if deterministic:
            mean,log_std = self.forward(state)  # Returns deterministic action
            mean = torch.tanh(mean) * self.max_action
            return mean
        else:
            action, _ = self.sample_action(state)
            return action