from __future__ import annotations

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np

from comprl.client import Agent, launch_client

from TD3.td3 import TD3Agent
    
    
def init_agent(args: list[str]) -> Agent:

    env = h_env.HockeyEnv()  # normal environment by default

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    max_action = env.action_space.high[0]

    agent = TD3Agent(
        state_dim=obs_dim,
        action_dim=action_dim,
        max_action=max_action,
        model_kwargs={
            "hidden_size": 256, 
            "num_hidden_layers": 2
        },
        action_range=None,       # if you had a custom sampler, pass it here
        discount=0.99,
        tau=0.005,
        buffer_size=1000000,
        actor_lr=2e-4,
        critic_lr=2e-4,
        batch_size=128,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        train_iter=1,
        exploration_noise=0.1,
        use_priority=False,      # If you want to use Prioritized Replay
    )

    agent.load_weights(path_actor="results/td3_selfplay_add_sac_5M/weights_1000000/actor.pth", path_critic="results/td3_selfplay_add_sac_5M/weights_1000000/critic.pth")

    return agent
    

def main() -> None:
    
    launch_client(init_agent)


if __name__ == "__main__":
    main()