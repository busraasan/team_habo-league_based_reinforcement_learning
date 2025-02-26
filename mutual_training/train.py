# train_td3_hockey_env.py

import sys
import os
sys.path.append("..")  # Adjust if your modules are one folder up
import numpy as np
import torch
import random
import matplotlib
import matplotlib.pyplot as plt

# Hockey environment
import hockey.hockey_env as h_env
from hockey.hockey_env import HockeyEnv

# TD3 modules (assuming you have them in these files)
from TD3.td3 import TD3Agent         # Or wherever your TD3Agent is defined
from TD3.td3_trainer import TD3Trainer     # Or wherever your TD3Trainer is defined

import argparse

# write argeparse
parser = argparse.ArgumentParser()
parser.add_argument("--opponent", type=str, default="strong", choices=["weak", "strong", "TD3", "SAC"])
parser.add_argument("--model", type=str, default="TD3", choices=["TD3", "SAC"])
parser.add_argument("--add_sac", type=bool, default=False, choices=[True, False])
parser.add_argument("--save_path", type=str, default="results")

args = parser.parse_args()

torch.set_num_threads(4)  # Adjust based on your CPU

# ---------------------------
# Seeding utility (optional)
# ---------------------------
def seed_everything(seed: int = 21412):
    import os
    import numpy as np
    import torch
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_data(obs, game_num):
    # Save data
    obs = np.array(obs)
    np.save('obs_'+game_num+'.npy', obs)

# If GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_TD3():

    seed_everything(21412)

    env = HockeyEnv()  # normal environment by default

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    max_action = env.action_space.high[0]

    td3_agent = TD3Agent(
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
        device=device
    )

    trainer = TD3Trainer(
        config={"opponent": args.opponent,
                "continue_training": True,
                "state_dim":obs_dim,
                "action_dim":action_dim,
                "max_action":max_action,
                "device":"cpu",
                "num_envs":1,
                "selfplay":True},  # or "strong"
        logger=None
    )

    result_path = args.save_path
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(result_path, "weights"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "videos_strong"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "videos_sac"), exist_ok=True)

    max_steps = 5000000
    eval_step = 50000

    critic_losses, actor_losses, win_rates = trainer.train_td3_selfplay(
        TD3_agent=td3_agent,
        env=env,
        max_steps=max_steps,
        eval_step=eval_step,
        path = result_path,
        save_step=500000,
        add_sac=args.add_sac
    )
 
    td3_agent.save_weights(os.path.join(result_path, "weights"))

if __name__ == "__main__":
    train_TD3()
