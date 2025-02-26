import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
from TD3.actor import TD3Actor
from TD3.critic import TD3Critic
from TD3.replay_buffer import PrioritizedReplayBuffer, ReplayMemory
import os 
import torch.nn.functional as F
from comprl.client import Agent

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class TD3Agent(Agent):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 max_action, 
                 model_kwargs={}, 
                 action_range=None, 
                 **userconfig):
        super(TD3Agent, self).__init__()
        """
        TD3 Agent similar in structure to the DQNAgent, but for continuous actions.
        
        Args:
            state_dim (int): Dimensionality of state/observation space.
            action_dim (int): Dimensionality of the action space.
            max_action (float): Maximum absolute value for each action dimension.
            model_kwargs (dict): Extra config for Actor/Critic networks (e.g., hidden sizes).
            action_range (function or object): If you need custom action sampling/exploration.
            userconfig (dict): Additional hyperparameters.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default Config and Update with User Config
        self._config = {
            "discount": 0.99,         # Gamma
            "tau": 0.005,            # Soft update coefficient
            "buffer_size": 1000000,  # Replay buffer capacity
            "batch_size": 256,       
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "policy_noise": 0.2,     # Noise added to target actions
            "noise_clip": 0.5,       # Clipping range of noise
            "policy_delay": 2,       # Delayed actor updates
            "train_iter": 1,         # Gradient steps per update call
            "exploration_noise": 0.1,# Std for adding noise to actions at inference
            "use_priority": True,   # If using PrioritizedReplayBuffer
            "update_target_every": 1,# For demonstration if you want a 'hard' or 'soft' combo
        }
        self._config.update(userconfig)

        self.gamma = self._config["discount"]
        self.tau = self._config["tau"]
        self.policy_noise = self._config["policy_noise"]
        self.noise_clip = self._config["noise_clip"]
        self.policy_delay = self._config["policy_delay"]
        self.train_iter = self._config["train_iter"]
        self.batch_size = self._config["batch_size"]
        self.exploration_noise = self._config["exploration_noise"]
        
        self.action_range = action_range  
        
        # 3. Replay Buffer (Prioritized or Regular)
        if self._config["use_priority"]:
            self.replay_buffer = PrioritizedReplayBuffer(self._config["buffer_size"])
        else:
            self.replay_buffer = ReplayMemory(self._config["buffer_size"])  

        # 4. Initialize Actor and Critic (plus Target Networks)
        self.actor = TD3Actor(model_kwargs, state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.critic = TD3Critic(model_kwargs, state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # 5. Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self._config["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self._config["critic_lr"])

        # 6. Tracking
        self.max_action = max_action
        self.current_train_iter = 0  # For optional scheduling or logs

    def get_step(self, observation: list[float]) -> list[float]:
        action = self.act(observation).clip(-1,1).tolist()
        return action
    
    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
            text_result = "won" if result else "lost"
            print(
                f"game ended: {text_result} with my score: "
                f"{stats[0]} against the opponent with score: {stats[1]}"
            )

    #  Select action with exploration noise
    def act(self, state, add_noise=True):
        """
        Given the current state, return a continuous action.
        Args:
            state (np.array): Current environment observation.
            add_noise (bool): Whether to add exploration noise.
        Returns:
            np.array: Continuous action clipped to [-max_action, max_action].
        """
        # Convert state to torch
        state_t = torch.FloatTensor(state).to(self.device)
        
        # Actor forward
        action = self.actor.forward(state_t)
        action = action.detach().cpu().numpy()

        # Optional Exploration Noise
        if add_noise: # for training not for evaluation
            action += np.random.normal(0, self.exploration_noise, size=action.shape)

        # Clip action to valid range
        action = np.clip(action, -self.max_action, self.max_action)

        return action

    # store_transition: Add to replay buffer
    def store_transition(self, *args):
        self.replay_buffer.add_transition(Transition(*args))

    # Train the critic(s) every step, and the actor occasionally
    def update(self):
        """
        Perform one or more gradient update steps on the actor and critic networks.
        """
        self.current_train_iter += 1
        total_critic_loss = 0.0
        total_actor_loss = 0.0

        # Optional: Hard update every 'update_target_every' if you want a DQN style
        # but typically in TD3 we do soft updates every iteration.
        if self.current_train_iter % self._config["update_target_every"] == 0:
            pass  # You could do a "hard update" or just rely on soft updates below

        for _ in range(self.train_iter):
            # Sample from replay buffer
            if self._config["use_priority"]:
                transitions, weights, indices = self.replay_buffer.sample(self.batch_size)
                weights_t = torch.FloatTensor(weights).unsqueeze(-1).to(self.device)
            else:
                transitions = self.replay_buffer.sample(self.batch_size)

            # Convert to tensors
            batch = Transition(*zip(*transitions))

            state = torch.FloatTensor(batch.state).to(self.device)
            action = torch.FloatTensor(batch.action).to(self.device)
            reward = torch.FloatTensor(batch.reward).unsqueeze(-1).to(self.device)
            next_state = torch.FloatTensor(batch.next_state).to(self.device)
            done = torch.FloatTensor(batch.done).unsqueeze(-1).to(self.device)

            # 1) Critic Update
            # Target policy smoothing
            with torch.no_grad():
                # Target action with clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

                # Compute target Q-value (twin Q-networks)
                q1_target, q2_target = self.critic_target(next_state, next_action, return_all=True)
                q_target_min = torch.min(q1_target, q2_target)
                y = reward + self.gamma * (1.0 - done) * q_target_min

            # Get current Q estimates
            q1_curr, q2_curr = self.critic(state, action)

            # Critic loss (Huber or MSE)
            critic_loss_1 = F.mse_loss(q1_curr, y)
            critic_loss_2 = F.mse_loss(q2_curr, y)
            critic_loss = critic_loss_1 + critic_loss_2

            # If using PER, scale the loss by importance sampling weights
            if self._config["use_priority"]:
                critic_loss = critic_loss * weights_t.mean()

            # Optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            total_critic_loss += critic_loss.item()

            # 2) Delayed Actor Update
            if self.current_train_iter % self.policy_delay == 0: 
                # Actor wants to maximize Q1(s, actor(s))
                actor_loss = -self.critic(state, self.actor(state), return_all=False).mean() # we update actor using the q1 value
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                total_actor_loss += actor_loss.item()

                # 3) Soft Update Target Networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # 4) Update Priorities in PER if used
            if self._config["use_priority"]:
                # Compute TD error for priority update
                with torch.no_grad():
                    td_error = torch.abs(q1_curr - y).cpu().numpy() + 1e-6
                self.replay_buffer.update_priorities(indices, td_error)

        return {
            "critic_loss": total_critic_loss / self.train_iter,
            "actor_loss": total_actor_loss / max(1, (self.train_iter // self.policy_delay)),
        }

	# Save actor & critic
    def save_weights(self, path):
        
        os.makedirs(path, exist_ok=True)
        path_actor = path + "/actor.pth"
        path_critic = path + "/critic.pth"
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic.state_dict(), path_critic)

	# Load pre-trained weights
    def load_weights(self, path_actor, path_critic):
        self.actor.load_state_dict(torch.load(path_actor, map_location=self.device))
        self.critic.load_state_dict(torch.load(path_critic, map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.set_eval()

	# Switch networks to eval mode
    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
