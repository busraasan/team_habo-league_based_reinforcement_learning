
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hockey.hockey_env as h_env
from sac import SAC,train_two_agents_from_pool,run_tournament
import gym
import torch 
from gym.vector import SyncVectorEnv
import time 
import matplotlib.pyplot as plt
import random 

# read the agent dirs in a given folder 

def get_agent_dirs(folder):
	agent_dirs = []
	for root, dirs, files in os.walk(folder):
		for dir in dirs:
			if dir.startswith("agent"):
				agent_dirs.append(os.path.join(root, dir))
	return agent_dirs

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
eval_env = h_env.HockeyEnv(mode = h_env.Mode.NORMAL)

device = "cpu"


# agent_weights = [
# 	"./results/2025-02-22_20-36-39/agent_0",
# 	"./results/2025-02-22_20-36-39/agent_1",
# 	"./results/2025-02-22_20-36-39/agent_2",
# 	"./results/2025-02-22_20-36-39/agent_3",
# 	"./results/2025-02-22_20-36-39/agent_2 2.250k",
# 	"./results/2025-02-22_20-36-39/agent_1 2.250k",
# ]

agent_weights = [
	"./results/2025-02-24_09-20-22_pool_train/agent_6_1800000",
	"./results/2025-02-24_09-20-22_pool_train/agent_6_2300000",
	"./results/2025-02-24_09-20-22_pool_train/agent_6_1700000",
	"./results/2025-02-24_09-20-22_pool_train/agent_3_1700000",
	"./results/2025-02-24_09-20-22_pool_train/agent_1_2300000",
	"./results/2025-02-24_09-20-22_pool_train/agent_7_1800000",
	"./results/2025-02-24_09-20-22_pool_train/agent_4_2200000",
	"./results/2025-02-24_09-20-22_pool_train/agent_4_2300000",
	"./results/2025-02-24_16-02-53_pool_train/agent_1_500000",
]

agent_weights += get_agent_dirs("./results/2025-02-24_16-02-53_pool_train")
# agent_weights = [dir_ for dir_ in agent_weights if ("1000" in dir_) or ("1100" in dir_)  or ("900" in dir_) or ("8000" in dir_) or ("7000" in dir_) or ("6000" in dir_) or ("5000" in dir_) or ("12000" in dir_)] 
agent_weights = [dir_ for dir_ in agent_weights if ("12000" in dir_) or ("13000" in dir_) or ("11000" in dir_) or ("14000" in dir_) or ("15000" in dir_)] 
agent_weights.append("./results/2025-02-22_20-36-39/agent_2")


# agent_weights.append("./results/2025-02-24_09-20-22_pool_train/agent_0_1600000")
# agent_weights.append("./results/2025-02-24_09-20-22_pool_train/agent_6_1700000")

agent_pool = []
number_of_agents = len(agent_weights)

for i in range(number_of_agents):
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
	# agent = add_noise(agent, 0.02)
	agent_pool.append(agent)


random.shuffle(agent_pool,random.random)
 

# def run_tournament(agent_pool, eval_env, num_episodes_per_match=5):

run_tournament(agent_pool,agent_weights, eval_env, num_episodes_per_match=200)






# ./results/2025-02-24_09-20-22_pool_train/agent_0_1600000


# ./results/2025-02-24_16-02-53_pool_train/agent_5_1400000



