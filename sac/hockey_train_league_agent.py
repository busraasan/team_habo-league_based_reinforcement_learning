import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hockey_sac.hockey_env as h_env_train
import hockey.hockey_env as h_env_eval
from TD3.td3 import TD3Agent

from sac import SAC,train_two_agents_from_pool
import gym
import torch 
from gym.vector import SyncVectorEnv
import time 
import matplotlib.pyplot as plt
import random 


def add_noise(model, noise_std):
	for param in model.actor.parameters():
		if random.random() < 0.2:
			param.data += torch.normal(0, noise_std, param.data.shape).to(param.data.device)
	for param in model.critic.parameters():
		if random.random() < 0.2:
			param.data += torch.normal(0, noise_std, param.data.shape).to(param.data.device)
   
	return model


torch.set_num_threads(4)  # Adjust based on your CPU

# envs = SyncVectorEnv([lambda: HockeyEnv() for _ in range(num_envs)])
# envs = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
# eval_env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)

mode = "NORMAL"

train_env = h_env_train.HockeyEnv(mode = h_env_train.Mode.NORMAL)
eval_env = h_env_eval.HockeyEnv(mode = h_env_eval.Mode.NORMAL)


device = "cpu"

agent_weights = [
	"./results/2025-02-24_09-20-22_pool_train/agent_6_1800000",
	"./results/2025-02-24_09-20-22_pool_train/agent_6_2300000",
	"./results/2025-02-24_09-20-22_pool_train/agent_6_1700000",
	"./results/2025-02-24_09-20-22_pool_train/agent_3_1700000",
	"./results/2025-02-24_09-20-22_pool_train/agent_1_2300000",
	"./results/2025-02-24_09-20-22_pool_train/agent_7_1800000",
	"./results/2025-02-24_09-20-22_pool_train/agent_4_2200000",
	"./results/2025-02-24_09-20-22_pool_train/agent_4_2300000",
]




agent_pool = []
for i in range(len(agent_weights)):
	agent = SAC(
		state_dim=eval_env.observation_space.shape[0],
		action_dim=eval_env.action_space.shape[0] // 2,
		max_action=eval_env.action_space.high[0],
		device=device,
		hyperparam_yaml="hyperparams.yaml",
		num_envs=1
	)
	agent.load_model(agent_weights[i])
	# agent.replay_buffer.clear()
	# agent = add_noise(agent, 0.01)
	agent_pool.append(agent)




path = f"results/{time.strftime('%Y-%m-%d_%H-%M-%S')}_pool_train"


pick_possibilties = [1.0/(len(agent_pool))]*(len(agent_pool))
print(pick_possibilties)

results,agent_pool_final = train_two_agents_from_pool(
	agent_pool,
	pick_possibilties,
	env = train_env,
	eval_env=eval_env,
	path=path,
	
)


for i,agent in enumerate(agent_pool_final):
	agent.save_model(f"{path}/weights_{i}")
 

