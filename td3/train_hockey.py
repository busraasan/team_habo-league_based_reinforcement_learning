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
from td3 import TD3Agent         # Or wherever your TD3Agent is defined
from td3_trainer import TD3Trainer     # Or wherever your TD3Trainer is defined

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

# If GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Set seed
    seed_everything(21412)

    env = HockeyEnv()  # normal environment by default

    obs_dim = env.observation_space.shape[0]  # e.g., 18 or more
    action_dim = env.action_space.shape[0] // 2                            # We assume [dx, dy, rotation, shoot_flag]
    max_action = env.action_space.high[0]                          # Usually we clip actions in [-1, 1]

    # 4. Initialize TD3 Agent
    #    Adjust hyperparameters as needed
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

    # 5. Initialize TD3 Trainer
    #    Configure the opponent as "weak" or "strong", etc.
    trainer = TD3Trainer(
        config={"opponent": "strong"},  # or "strong"
        logger=None
    )

    # 6. Train
    result_path = "results"
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(result_path, "weights"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "videos"), exist_ok=True)
    max_steps = 1000000    # environment steps
    eval_step = 50000      # evaluate every 50 steps
    critic_losses, actor_losses, win_rates = trainer.train_td3(
        TD3_agent=td3_agent,
        env=env,
        max_steps=max_steps,
        eval_step=eval_step
    )

	# 7. Save model
 
    td3_agent.save_weights(os.path.join(result_path, "weights/td3_vs_strong_opponent_model.pth"))
    

if __name__ == "__main__":
    main()
