import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hockey.hockey_env as h_env
from sac import SAC
import gym
import torch 
from gym.vector import SyncVectorEnv
import time 
import matplotlib.pyplot as plt

torch.set_num_threads(2)  # Adjust based on your CPU


num_envs = 1  # Number of parallel environments
# envs = SyncVectorEnv([lambda: HockeyEnv() for _ in range(num_envs)])
# envs = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
# eval_env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)

envs = h_env.HockeyEnv()
eval_env = h_env.HockeyEnv()

sac_agent = SAC(
    state_dim=eval_env.observation_space.shape[0],
	action_dim=eval_env.action_space.shape[0] // 2,
	max_action=eval_env.action_space.high[0],
	device="cpu",
	hyperparam_yaml="hyperparams.yaml",
	num_envs=num_envs
)

path = f"results/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(path, exist_ok=True)

critic_loss, actor_loss, alpha_loss, rewards = sac_agent.train(envs,eval_env,player2 = h_env.BasicOpponent(weak=False),path = os.path.join(path,"videos"))


sac_agent.save_model(os.path.join(path, "weights"))

# Plotting

plt.figure()
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Training Rewards")
plt.savefig(f"{path}/rewards.png")
plt.show()

plt.figure()
plt.plot(critic_loss)
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("Critic Loss")
plt.savefig(f"{path}/critic_loss.png")
plt.show()

plt.figure()
plt.plot(actor_loss)
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("Actor Loss")
plt.savefig(f"{path}/actor_loss.png")
plt.show()

plt.figure()
plt.plot(alpha_loss)
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("Alpha Loss")
plt.savefig(f"{path}/alpha_loss.png")
plt.show()



