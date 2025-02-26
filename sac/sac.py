import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hockey_sac.hockey_env as h_env
import torch
import torch.nn.functional as F
import yaml
import random
import numpy as np
import cv2
from tqdm import tqdm
from gym.wrappers import RecordVideo

from actor import SacActor
from critic import SacCritic
from replay_buffer import ReplayBuffer
from comprl.client import Agent


class SAC(Agent):
	def __init__(self, state_dim, action_dim, max_action,device,num_envs, hyperparam_yaml="./hyperparams.yaml"):
		"""
		Initializes the SAC agent by loading hyperparameters.

		Args:
			state_dim (int): Dimension of the state space.
			action_dim (int): Dimension of the action space.
			hyperparam_yaml (str): Path to the YAML file containing hyperparameters.
		"""
		super(SAC, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action
		self.device = device
		self.hyperparams = self.read_hyperparams(hyperparam_yaml)
		self.num_envs = num_envs
		self.set_seed(self.hyperparams['seed'])

		# Initialize components (Replay Buffer, Actor, Critic, Optimizers, etc.)
		self.initialize_components()

	def read_hyperparams(self, hyperparam_yaml):
		"""
		Reads hyperparameters from a YAML file.

		Args:
			hyperparam_yaml (str): Path to the YAML file.

		Returns:
			dict: Dictionary of hyperparameters.
		"""
		with open(hyperparam_yaml, "r") as f:
			hyperparams = yaml.safe_load(f)
		return hyperparams

	def set_seed(self, seed):
		"""
		Sets the seed for reproducibility.

		Args:
			seed (int): The seed value.
		"""
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


	def initialize_components(self):
		"""
		Initializes necessary components for SAC (Replay Buffer, Actor, Critic, etc.).
		"""
		# Keep the elemenet in the cpu and move them to the device when needed.
		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.hyperparams['replay_buffer']['size'], self.hyperparams['replay_buffer']['device'],self.num_envs)

		# Initialize Actor
		self.actor = SacActor(self.hyperparams['actor'], self.state_dim, self.action_dim, self.max_action).to(self.device)

		self.critic = SacCritic(self.hyperparams['critic'], self.state_dim, self.action_dim).to(self.device)

		self.critic_target = SacCritic(self.hyperparams['critic'], self.state_dim, self.action_dim).to(self.device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_target.requires_grad_(False)


		if self.hyperparams['optim']["critic"]['type'] == "Adam":
			self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hyperparams['optim']["critic"]['lr'])
		else:
			raise NotImplementedError("Only Adam optimizer is supported for critic.")

		if self.hyperparams['optim']["actor"]['type'] == "Adam":
			self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hyperparams['optim']["actor"]['lr'])
		else:
			raise NotImplementedError("Only Adam optimizer is supported for actor.")

		self.target_entropy = -self.action_dim
  
		self.log_alpha = torch.Tensor([self.hyperparams['alpha']]).to(self.device)
		if self.hyperparams["auto_alpha"]:
			self.log_alpha.requires_grad = True
			if self.hyperparams["optim"]["alpha"]["type"] == "Adam":
				self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.hyperparams["optim"]["alpha"]["lr"])
			else:
				raise NotImplementedError("Only Adam optimizer is supported for alpha.")

		self.reward_normalizer = RewardNormalizer()
		self.obs_normalizer = ObsNormalizer(self.state_dim)
		


	def update_target_network(self):
		"""
		Updates the target Q-network using soft updates.
		"""
		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(self.hyperparams['tau'] * param.data + (1 - self.hyperparams['tau']) * target_param.data)


	def select_action(self, state,deterministic=False):
		"""
		Selects an action based on the policy.

		Args:
			state (np.array): The input state.
			deterministic (bool): Whether to select the deterministic action.

		Returns:
			np.array: The chosen action.
		"""

		if len(state.shape) > 1:
			bsize = state.shape[0]
		else:
			bsize = 1

		state = torch.FloatTensor(state.reshape(bsize,-1)).to(self.device)

		if deterministic:
			with torch.no_grad():
				action = self.actor.get_action(state,deterministic=True)
		else:
			with torch.no_grad():
				action = self.actor.get_action(state,deterministic=False)

		return action.cpu().data.numpy().flatten()


	def act(self, state):
		"""
		Selects an action based on the policy.

		Args:
			state (np.array): The input state.

		Returns:
			np.array: The chosen action.
		"""
		state = self.obs_normalizer.normalize(state)
		return self.select_action(state,deterministic=True)


	def one_step_train(self, batch_size):
		"""
		Performs one training step of SAC.

		Args:
			batch_size (int): The size of the training batch.

		Returns:
			dict: Training loss values.
		"""
		if len(self.replay_buffer) < batch_size:
			return None

		# Sample a batch from the replay buffer
		state, action, reward, next_state, done = self.replay_buffer.sample(batch_size, self.device)

		# Compute the target Q value
		with torch.no_grad():
			next_action, next_log_prob = self.actor.sample_action(next_state)
			q1_target, q2_target = self.critic_target(next_state, next_action)
			q_target = torch.min(q1_target, q2_target)
			target = reward + self.hyperparams['gamma'] * (1 - done) * (q_target - self.log_alpha.exp() * next_log_prob)

		# Update Q networks
		q1, q2 = self.critic(state, action)
		critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

		self.critic_optimizer.zero_grad() # clean prev gradients
		critic_loss.backward() # compute gradients

		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hyperparams['grad_clip'])
		self.critic_optimizer.step()

		# Update policy network
		new_action, log_prob = self.actor.sample_action(state)
		q1_new, q2_new = self.critic(state, new_action)
		q_new = torch.min(q1_new, q2_new)
		actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hyperparams['grad_clip'])
		self.actor_optimizer.step()

		# Update temperature parameter
		alpha_loss = 0.0
		if self.hyperparams["auto_alpha"]:
			alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()


		losses = {'Critic Loss': critic_loss.item(), 'Actor Loss': actor_loss.item(), 'Alpha Loss': alpha_loss.item()}
		return losses


	def train(self, env,eval_env,player2,path):
		"""
		Trains the SAC agent on the given environment.

		Args:
			env (gym.Env): The training environment.
		"""
		batch_size = self.hyperparams["train"]["batch_size"]
		max_steps = self.hyperparams["train"]["max_steps"]
		start_steps = self.hyperparams["train"]["start_steps"]

		target_update_interval = self.hyperparams["train"]["target_update_interval"]

		log_interval = self.hyperparams["train"]["log_interval"]
		eval_interval = self.hyperparams["train"]["eval_interval"]
		num_eval_episodes = self.hyperparams["train"]["num_eval_episodes"]

		critic_loss = []
		actor_loss = []
		alpha_loss = []
		rewards = []

		current_step = 0
		state,_ = env.reset()
		state.reshape(self.num_envs, -1)
		done = False

		
		tqdm_bar = tqdm(range(max_steps), desc="Training", unit="step")
		while True:
			if current_step > max_steps:
				break

			# env.render(mode="human")
			action1 = self.select_action(state).reshape(self.num_envs, -1)
			action1 = action1.reshape(-1)
			obs_agent2 = env.obs_agent_two()
			action2 = player2.act(obs_agent2)
			next_state, reward, done, _, _ = env.step(np.hstack([action1,action2]))
			self.obs_normalizer.update(next_state)
			next_state = self.obs_normalizer.normalize(next_state)
   
			# next_state, reward, done = next_state.reshape(self.num_envs, -1), reward.reshape(self.num_envs, -1), done.reshape(self.num_envs, -1)
			if np.isscalar(reward):
				reward,done = np.array([reward]), np.array([done])
	
			self.reward_normalizer.update(reward)
			reward = self.reward_normalizer.normalize(reward)
			self.replay_buffer.add(state, action1, reward, next_state, done)
			if current_step > start_steps:
				for i in range(self.num_envs):
					losses = self.one_step_train(batch_size)
					if losses is not None:
						critic_loss.append(losses['Critic Loss'])
						actor_loss.append(losses['Actor Loss'])
						alpha_loss.append(losses['Alpha Loss'])
					if current_step % target_update_interval == 0:
						self.update_target_network()
					if current_step % log_interval == 0 :
						if critic_loss:
							tqdm_bar.set_postfix({'Critic Loss': sum(critic_loss)/len(critic_loss), 'Actor Loss': sum(actor_loss)/len(actor_loss), 'Alpha Loss': sum(alpha_loss)/len(alpha_loss)})

			if self.num_envs > 1:
				for i in range(self.num_envs):
					if done[i]:
						state[i],_ = env.reset(i)
				else:
					state = next_state
			else:
				if done:
					state,_ = env.reset()
					self.obs_normalizer.update(state)
					state = self.obs_normalizer.normalize(state)
				else:
					state = next_state
	 
			current_step += self.num_envs
			tqdm_bar.update(self.num_envs)

			if current_step % eval_interval == 0:
				print("Starting Evaluation")
				rewards_step = self.eval(eval_env,player2, num_eval_episodes,current_step,video_dir = path)
				print(f"Step: {current_step}, Mean Reward: {sum(rewards_step)/len(rewards_step)}")
				rewards.append(sum(rewards_step)/len(rewards_step))

		tqdm_bar.close()
		return critic_loss, actor_loss, alpha_loss, rewards

	def eval(self, env, player2, num_episodes,current_step, video_dir="videos", log_path = "log.txt", fps=30):
		"""
		Evaluates the SAC agent and saves video recordings manually using OpenCV.

		Args:
			env (gym.Env): The environment.
			num_episodes (int): Number of evaluation episodes.
			video_dir (str): Directory to save videos.
			fps (int): Frames per second for the saved video.

		Returns:
			list: List of rewards for each episode.
		"""

		# Ensure video directory exists
		os.makedirs(video_dir, exist_ok=True)

		agent_winner = 0
		bot_winner = 0
		draw = 0

		rewards = []
		for i in range(num_episodes):
			reward = 0
			state, _ = env.reset()
			done = False
			step_num = 0
			frames = []

			while not is_terminal and step_num < 10000:
				# Render frame as an RGB array
				if i < 5:
					frame = env.render(mode = "rgb_array")
				if frame is not None:
					frames.append(frame)

				# Select action and step environment
				state = self.obs_normalizer.normalize(state)
				action1 = self.select_action(state,deterministic=True)
				action1 = action1.reshape(-1)
				obs_agent2 = env.obs_agent_two()
				action2 = player2.act(obs_agent2)
				next_state, r, done, truncated, info = env.step(np.hstack([action1,action2]))
				is_terminal = done or truncated
				reward += r
				state = next_state
				step_num += 1

			if info['winner'] == 1:
				agent_winner += 1
			elif info['winner'] == -1:
				bot_winner += 1
			else:
				draw +=1
	
			
			rewards.append(reward)
			if i < 5:
				video_path = os.path.join(video_dir, f"{current_step}_episode_{i+1}.mp4")
				save_video(frames, video_path, fps)
		# print the results as %
		print(f"Agent Wins: {agent_winner/num_episodes*100}%, Bot Wins: {bot_winner/num_episodes*100}%, Draws: {draw/num_episodes*100}%")
		# log the results
		with open(log_path, "a") as f:
			f.write(f"Step {current_step}: Agent Wins: {agent_winner/num_episodes*100}%, Bot Wins: {bot_winner/num_episodes*100}%, Draws: {draw/num_episodes*100}%\n")
  
		return rewards,agent_winner/num_episodes*100,bot_winner/num_episodes*100,draw/num_episodes*100

	def act(self, state):
		"""
		Selects an action based on the policy. for the evaluation

		Args:
			state (np.array): The input state.

		Returns:
			np.array: The chosen action.
		"""
		state = self.obs_normalizer.normalize(state)
		return self.select_action(state,deterministic=True).reshape(-1)


	def save_model(self, model_dir):
		"""
		Saves the model parameters.

		Args:
			model_dir (str): Directory to save the model.
		"""
		os.makedirs(model_dir, exist_ok=True)
		torch.save(self.actor.state_dict(), os.path.join(model_dir, "actor.pth"))
		torch.save(self.critic.state_dict(), os.path.join(model_dir, "critic.pth"))
		torch.save(self.critic_target.state_dict(), os.path.join(model_dir, "critic_target.pth"))

		self.replay_buffer.save(os.path.join(model_dir, "replay_buffer.pth"))
  
		self.obs_normalizer.save_model(model_dir)
		self.reward_normalizer.save_model(model_dir)

		if self.hyperparams["auto_alpha"]:
			torch.save(self.log_alpha, os.path.join(model_dir, "log_alpha.pth"))
	
	def load_model(self,model_dir):
		"""
		Loads the model parameters.

		Args:
			model_dir (str): Directory to load the model from.
		"""
		self.actor.load_state_dict(torch.load(os.path.join(model_dir, "actor.pth")))
		self.critic.load_state_dict(torch.load(os.path.join(model_dir, "critic.pth")))
		self.critic_target.load_state_dict(torch.load(os.path.join(model_dir, "critic_target.pth")))

		self.replay_buffer.load(os.path.join(model_dir, "replay_buffer.pth"))
  
		self.obs_normalizer.load_model(model_dir)
		self.reward_normalizer.load_model(model_dir)

		if self.hyperparams["auto_alpha"]:
			self.log_alpha = torch.load(os.path.join(model_dir, "log_alpha.pth"))
   
   
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
		

def save_video(frames, filename, fps=30):
	""" Saves a list of frames as a video using OpenCV. """
	height, width, _ = frames[0].shape
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use MP4 format
	out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

	for frame in frames:
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (OpenCV format)
		out.write(frame)

	out.release()
 


class ObsNormalizer:
	def __init__(self, obs_dim, epsilon=1e-8):
		"""
		Normalizes observations element-wise.
		Args:
			obs_dim (int or tuple): Shape of the observation space.
			epsilon (float): Small value to avoid division by zero.
		"""
		self.mean = np.zeros(obs_dim, dtype=np.float32)
		self.var = np.ones(obs_dim, dtype=np.float32)
		self.count = 1e-4  # To prevent division by zero
		self.epsilon = epsilon

	def update(self, obs):
		""" Update mean and variance for each element in the observation vector. """
		obs = np.array(obs, dtype=np.float32)  # Ensure correct format
		self.count += 1
		delta = obs - self.mean
		self.mean += delta / self.count
		delta2 = obs - self.mean
		self.var += delta * delta2  # Online variance update

	def normalize(self, obs):
		""" Normalize each element of the observation vector. """
		return (obs - self.mean) / (np.sqrt(self.var / self.count) + self.epsilon)
	
	def denormalize(self, obs):
		""" Denormalize each element of the observation vector. """
		return obs * (np.sqrt(self.var / self.count) + self.epsilon) + self.mean

 
	def save_model(self, model_dir):
		np.save(os.path.join(model_dir, "obs_mean.npy"), self.mean)
		np.save(os.path.join(model_dir, "obs_var.npy"), self.var)
		np.save(os.path.join(model_dir, "obs_count.npy"), self.count)

	def load_model(self, model_dir):
		self.mean = np.load(os.path.join(model_dir, "obs_mean.npy"))
		self.var = np.load(os.path.join(model_dir, "obs_var.npy"))
		self.count = np.load(os.path.join(model_dir, "obs_count.npy"))


class RewardNormalizer:
	def __init__(self, epsilon=1e-8):
		self.mean = 0.0
		self.var = 1.0
		self.count = 1e-4  # Prevents division by zero
		self.epsilon = epsilon

	def update(self, reward):
		""" Update mean and variance for a single reward using Welford's Algorithm. """
		reward = float(reward)  # Ensure it's a single scalar
		self.count += 1
		delta = reward - self.mean
		self.mean += delta / self.count
		delta2 = reward - self.mean
		self.var += delta * delta2  # Online variance update

	def normalize(self, reward):
		""" Normalize rewards using running mean and variance. """
		return (reward - self.mean) / (np.sqrt(self.var / self.count) + self.epsilon)

	def save_model(self, model_dir):
		np.save(os.path.join(model_dir, "reward_mean.npy"), self.mean)
		np.save(os.path.join(model_dir, "reward_var.npy"), self.var)
		np.save(os.path.join(model_dir, "reward_count.npy"), self.count)	
  
	def load_model(self,model_dir):
		self.mean = np.load(os.path.join(model_dir, "reward_mean.npy"))
		self.var = np.load(os.path.join(model_dir, "reward_var.npy"))
		self.count = np.load(os.path.join(model_dir, "reward_count.npy"))


def compute_behavior_descriptor(agent, fixed_opponent, env, num_steps=100):
	"""
	Runs a short rollout in a two-player environment where:
	  - agent controls player1
	  - fixed_opponent controls player2  (hand-coded or stable)
	Returns an "action histogram" (or any other descriptor).
	"""

	obs1, _ = env.reset()
	obs2 = env.obs_agent_two()

	# Optional: normalize if agent or env expects that
	# But typically for the descriptor, we can keep it raw or do what agent does in normal training.
	if hasattr(agent, "obs_normalizer"):
		obs1 = agent.obs_normalizer.normalize(obs1)

	# If the opponent is also an SAC or something, we might or might not do normalization.
	# For a hand-coded opponent, we just call .act(obs).

	action_hist = np.zeros(agent.action_dim, dtype=np.float32)

	for _ in range(num_steps):
		# agent's action
		action1 = agent.select_action(obs1, deterministic=True)

		# opponent's action
		if hasattr(fixed_opponent, "act"):
			# if it's a BasicOpponent or some class with .act()
			action2 = fixed_opponent.act(obs2)
		else:
			# or if it's a random or rule-based approach
			action2 = fixed_opponent.select_action(obs2, deterministic=True)

		# track some measure of agent's action
		action_hist += (action1 > 0).astype(np.float32)

		# Step environment
		joint_action = np.hstack([action1, action2])
		next_obs1, _, done, truncated, _ = env.step(joint_action)
		next_obs2 = env.obs_agent_two()

		# If we want to keep normalizing:
		if hasattr(agent, "obs_normalizer"):
			next_obs1 = agent.obs_normalizer.normalize(next_obs1)

		obs1 = next_obs1
		obs2 = next_obs2

		if done or truncated:
			obs1, _ = env.reset()
			obs2 = env.obs_agent_two()
			if hasattr(agent, "obs_normalizer"):
				obs1 = agent.obs_normalizer.normalize(obs1)

	# Normalize the histogram
	action_hist /= float(num_steps)
	return action_hist


def compute_diversity_bonus(agent, agent_pool, alpha=1.0):
	"""
	Compute how different this agent's descriptor is from the rest of the pool.
	A simple approach: find average L2 distance from all other agents' descriptors.
	Returns a scalar "diversity bonus".
	"""
	if getattr(agent, "descriptor", None) is None:
		# If we have no descriptor yet, no bonus
		return 0.0

	distances = []
	for other in agent_pool:
		# Skip itself or any opponents with no descriptor
		if other is agent or getattr(other, "descriptor", None) is None:
			continue
		# measure L2 distance
		dist = np.linalg.norm(agent.descriptor - other.descriptor)
		distances.append(dist)
	if len(distances) == 0:
		return 0.0
	avg_dist = sum(distances) / len(distances)
	return alpha * avg_dist  # scale by alpha if you like


def train_two_agents_from_pool(
	agent_pool,
	pick_possibilities,
	env,
	eval_env,
	path="videos",
	max_steps=10_000_000,
	start_steps=50_000,
	eval_interval=100_000,
	log_interval=5000,
	num_eval_episodes=200,
):
	os.makedirs(path, exist_ok=True)
	log_path = path + "/log.txt"

	current_step = 0
	obs1, info1 = env.reset()
	obs2 = env.obs_agent_two()
	number_of_agents = len(agent_pool)
	done = True
	truncated = False  # Make sure truncated is defined

	critic_loss_1, actor_loss_1, alpha_loss_1 = [], [], []
	critic_loss_2, actor_loss_2, alpha_loss_2 = [], [], []
	eval_rewards = []

	pbar = tqdm(range(max_steps), desc="Training", unit="step")
	alternative_opponent = h_env.BasicOpponent(weak=False)

	one_episode_actions_agent1 = []
	one_episode_actions_agent2 = []
	
	while current_step < max_steps:
		if done or truncated:
			# Pick random 2 agents from pool
			agent1 = np.random.choice(agent_pool, p=pick_possibilities)
			agent2 = np.random.choice(agent_pool, p=pick_possibilities)
			
			# Make sure they're different
			rand_number = random.random()
			use_alt_opp_for_player1 = False
			use_alt_opp_for_player2 = False
			if rand_number < 0.05:
				agent1 = alternative_opponent
				use_alt_opp_for_player1 = True
			elif rand_number < 0.1:
				agent2 = alternative_opponent
				use_alt_opp_for_player2 = True

			while agent1 == agent2:
				agent2 = np.random.choice(agent_pool, p=pick_possibilities)
   
			obs1, info1 = env.reset()
			done = False
			truncated = False

			if not use_alt_opp_for_player1 and hasattr(agent1, "obs_normalizer"):
				agent1.obs_normalizer.update(obs1)
				obs1 = agent1.obs_normalizer.normalize(obs1)
			if not use_alt_opp_for_player2 and hasattr(agent2, "obs_normalizer"):
				obs2 = env.obs_agent_two()
				agent2.obs_normalizer.update(obs2)
				obs2 = agent2.obs_normalizer.normalize(obs2)

		# env.render(mode="human")
		action1 = agent1.select_action(obs1) 
		one_episode_actions_agent1.append(action1)
		action2 = agent2.select_action(obs2) 
		one_episode_actions_agent2.append(action2)
  
		next_obs, r1, done, truncated, info = env.step(np.hstack([action1, action2]))
		next_obs2 = env.obs_agent_two()
		if not use_alt_opp_for_player1 and hasattr(agent1, "obs_normalizer"):
			agent1.obs_normalizer.update(next_obs)
			next_obs = agent1.obs_normalizer.normalize(next_obs)
		if not use_alt_opp_for_player2 and hasattr(agent2, "obs_normalizer"):
			agent2.obs_normalizer.update(next_obs2)
			next_obs2 = agent2.obs_normalizer.normalize(next_obs2)

		info2 = env.get_info_agent_two()
		r2 = env.get_reward_agent_two(info2)

		is_terminal = done or truncated

		# ~~~ DIVERSITY: We'll add the bonus only at the final step of the episode. ~~~
		# So if is_terminal == True, we compute a diversity bonus for each agent
		diversity_bonus_1 = 0.0
		diversity_bonus_2 = 0.0

		if is_terminal:
			# Only if these are real agents, not alt opponent
			if not use_alt_opp_for_player1:
				# 1) Update or compute agent1 descriptor
				agent1.descriptor = compute_behavior_descriptor(agent1,alternative_opponent, eval_env, num_steps=200)
				# 2) Calculate how different agent1 is from the rest
				diversity_bonus_1 = compute_diversity_bonus(agent1, agent_pool, alpha=1.0)

			if not use_alt_opp_for_player2:
				agent2.descriptor = compute_behavior_descriptor(agent2,alternative_opponent, eval_env, num_steps=200)
				diversity_bonus_2 = compute_diversity_bonus(agent2, agent_pool, alpha=1.0)

   
			if not use_alt_opp_for_player1:
				agent1_action_hist_first_half = np.array(one_episode_actions_agent1[:len(one_episode_actions_agent1)//2])
				agent1_action_hist_second_half = np.array(one_episode_actions_agent1[len(one_episode_actions_agent1)//2:])
				agent1_action_hist_first_half = np.sum(agent1_action_hist_first_half, axis=0) / len(agent1_action_hist_first_half)
				agent1_action_hist_second_half = np.sum(agent1_action_hist_second_half, axis=0) / len(agent1_action_hist_second_half)
	
				# if the actions are varying, the norm will be high so the bonus will be highh
				diversity_bonus_1 += np.linalg.norm(agent1_action_hist_first_half - agent1_action_hist_second_half) 
	
			if not use_alt_opp_for_player2:	
				agent2_action_hist_first_half = np.array(one_episode_actions_agent2[:len(one_episode_actions_agent2)//2])
				agent2_action_hist_second_half = np.array(one_episode_actions_agent2[len(one_episode_actions_agent2)//2:])
				agent2_action_hist_first_half = np.sum(agent2_action_hist_first_half, axis=0) / len(agent2_action_hist_first_half)
				agent2_action_hist_second_half = np.sum(agent2_action_hist_second_half, axis=0) / len(agent2_action_hist_second_half)
				diversity_bonus_2 += np.linalg.norm(agent2_action_hist_first_half - agent2_action_hist_second_half) 
	

   
		# If they're real agents, add to replay with reward + bonus (only if terminal)
		if not use_alt_opp_for_player1 and hasattr(agent1, "reward_normalizer"):
			agent1.reward_normalizer.update(r1 + diversity_bonus_1)
			r1_norm = agent1.reward_normalizer.normalize(r1 + diversity_bonus_1)
			agent1.replay_buffer.add(obs1, action1, r1_norm, next_obs, float(is_terminal))

		if not use_alt_opp_for_player2 and hasattr(agent2, "reward_normalizer"):
			agent2.reward_normalizer.update(r2 + diversity_bonus_2)
			r2_norm = agent2.reward_normalizer.normalize(r2 + diversity_bonus_2)
			agent2.replay_buffer.add(obs2, action2, r2_norm, next_obs2, float(is_terminal))

		obs1 = next_obs
		obs2 = next_obs2
		current_step += 1
		pbar.update(1)

		# ~~~ TRAINING STEP ~~~
		if current_step > start_steps:
			if not use_alt_opp_for_player1:
				loss1 = agent1.one_step_train(agent1.hyperparams["train"]["batch_size"])
				if loss1 is not None:
					critic_loss_1.append(loss1["Critic Loss"])
					actor_loss_1.append(loss1["Actor Loss"])
					alpha_loss_1.append(loss1["Alpha Loss"])
				if current_step % agent1.hyperparams["train"]["target_update_interval"] == 0:
					agent1.update_target_network()

			if not use_alt_opp_for_player2:
				loss2 = agent2.one_step_train(agent2.hyperparams["train"]["batch_size"])
				if loss2 is not None:
					critic_loss_2.append(loss2["Critic Loss"])
					actor_loss_2.append(loss2["Actor Loss"])
					alpha_loss_2.append(loss2["Alpha Loss"])
				if current_step % agent2.hyperparams["train"]["target_update_interval"] == 0:
					agent2.update_target_network()

		# Logging
		if current_step % log_interval == 0:
			if critic_loss_1 and critic_loss_2:
				pbar.set_postfix({
					"Critic Loss 1": sum(critic_loss_1) / len(critic_loss_1),
					"Actor Loss 1": sum(actor_loss_1) / len(actor_loss_1),
					"Alpha Loss 1": sum(alpha_loss_1) / len(alpha_loss_1),
					"Critic Loss 2": sum(critic_loss_2) / len(critic_loss_2),
					"Actor Loss 2": sum(actor_loss_2) / len(actor_loss_2),
					"Alpha Loss 2": sum(alpha_loss_2) / len(alpha_loss_2),
				})
				with open(log_path, "a") as f:
					f.write(f"Step {current_step}: Critic Loss 1 = {sum(critic_loss_1) / len(critic_loss_1):.2f} ")
					f.write(f" Actor Loss 1 = {sum(actor_loss_1) / len(actor_loss_1):.2f} ")
					f.write(f" Alpha Loss 1 = {sum(alpha_loss_1) / len(alpha_loss_1):.2f}\n")
	 
					f.write(f"Step {current_step}: Critic Loss 2 = {sum(critic_loss_2) / len(critic_loss_2):.2f}")
					f.write(f" {current_step}: Actor Loss 2 = {sum(actor_loss_2) / len(actor_loss_2):.2f}")
					f.write(f" {current_step}: Alpha Loss 2 = {sum(alpha_loss_2) / len(alpha_loss_2):.2f}\n")

		# Evaluate
		if current_step % eval_interval == 0 and current_step > start_steps:
			agent_names = [f"Agent {i}" for i in range(number_of_agents)]
			os.makedirs(path + f"/{current_step}", exist_ok=True)
			_,_, stats = run_tournament(agent_pool, agent_names, eval_env, num_episodes_per_match=50, video_dir=path + f"/{current_step}")
   
			with open(log_path, "a") as f:
				f.write(f"Step {current_step}: Tournament Results\n")
				for name, s in stats.items():
					f.write(f"{name}: {s}\n")

			# Save models
			for i in range(number_of_agents):
				agent_pool[i].save_model(f"{path}/agent_{i}_{current_step}")
	

	pbar.close()

	return {
		"critic_loss_1": critic_loss_1,
		"actor_loss_1": actor_loss_1,
		"alpha_loss_1": alpha_loss_1,
		"critic_loss_2": critic_loss_2,
		"actor_loss_2": actor_loss_2,
		"alpha_loss_2": alpha_loss_2,
		"eval_rewards": eval_rewards,
	}, agent_pool





def evaluate_match(agent1, agent2, eval_env, num_episodes=5, max_steps=10000, record_video=False, video_dir=None):
	"""
	Runs a series of evaluation episodes between agent1 and agent2.
	Returns the number of wins for each agent.
	
	If record_video is True and video_dir is provided, the first 5 episodes are recorded as video.
	
	Assumes the environment returns an info dict with a 'winner' key,
	where 1 indicates agent1 win, -1 indicates agent2 win, and 0 is a draw.
	"""
	wins_agent1 = 0
	wins_agent2 = 0

	for ep in range(num_episodes):
		if record_video and video_dir is not None and ep < 5:
			frames = []  # list to store frames for video recording

		obs, _ = eval_env.reset()
		done = False
		step_count = 0

		while not done and step_count < max_steps:
			if record_video and video_dir is not None and ep < 5:
				frame = eval_env.render(mode="rgb_array")
				if frame is not None:
					frames.append(frame)
					
			# For evaluation, use deterministic policies:
			action1 = agent1.act(obs)  # assume act() returns a deterministic action
			obs_agent2 = eval_env.obs_agent_two()
			action2 = agent2.act(obs_agent2)
			
			joint_action = np.hstack([action1, action2])
			next_obs, reward, done, truncated, info = eval_env.step(joint_action)
			obs = next_obs
			step_count += 1

			if done or truncated:
				break

		# Check the outcome (if available)
		winner = info.get("winner", 0)
		if winner == 1:
			wins_agent1 += 1
		elif winner == -1:
			wins_agent2 += 1

		# Save video for this episode if required
		if record_video and video_dir is not None and ep < 5 and frames:
			filename = os.path.join(video_dir, f"final_match_episode_{ep+1}.mp4")
			save_video(frames, filename, fps=30)

	return wins_agent1, wins_agent2


def run_tournament(agent_pool, names, eval_env, num_episodes_per_match=5, video_dir="final_match_videos"):
	"""
	Runs a knockout tournament on the agent pool.
	
	For each round, agents are randomly paired. Each pair plays a match 
	(multiple evaluation episodes) and the agent with more wins advances.
	
	If there is an odd number of agents, one agent gets a bye.
	In the final round (last two agents), videos are recorded for the first 5 episodes.
	
	Also, match statistics are tracked for each agent (overall wins and losses).
	
	Returns:
		tuple: (final_winner, final_name, match_stats)
			where match_stats is a dict {agent_name: {"wins": total_wins, "losses": total_losses}}.
	"""
	# Initialize match statistics for each agent
	match_stats = {name: {"wins": 0, "losses": 0} for name in names}
	
	# Create a list of tuples: (agent, name)
	current_round_agents = list(zip(agent_pool, names))
	round_num = 1

	while len(current_round_agents) > 1:
		print(f"Round {round_num}: {len(current_round_agents)} agents")
		next_round_agents = []
		random.shuffle(current_round_agents)

		# If odd number, give a bye to one agent.
		if len(current_round_agents) % 2 == 1:
			bye_agent, bye_name = current_round_agents.pop()
			print(f"Agent {bye_name} gets a bye this round.")
			# Count a bye as winning all episodes for that round.
			match_stats[bye_name]["wins"] += num_episodes_per_match
			next_round_agents.append((bye_agent, bye_name))

		# For the final round, record videos in the match.
		record_video = (len(current_round_agents) == 2)
		if record_video:
			os.makedirs(video_dir, exist_ok=True)

		# Pair agents and run matches.
		for i in range(0, len(current_round_agents), 2):
			agent1, name1 = current_round_agents[i]
			agent2, name2 = current_round_agents[i+1]
			
			wins1, wins2 = evaluate_match(
				agent1, agent2, eval_env, num_episodes=num_episodes_per_match,
				record_video=record_video, video_dir=video_dir
			)
			print(f"Match: {name1} wins {wins1} - {name2} wins {wins2}")
			# Update match statistics.
			match_stats[name1]["wins"] += wins1
			match_stats[name1]["losses"] += wins2
			match_stats[name2]["wins"] += wins2
			match_stats[name2]["losses"] += wins1

			if wins1 >= wins2:
				winner = (agent1, name1)
			else:
				winner = (agent2, name2)
			print(f"Winner: {winner[1]}")
			next_round_agents.append(winner)
		current_round_agents = next_round_agents
		round_num += 1

	final_winner, final_name = current_round_agents[0]
	print(f"Final Winner: {final_name}")
	return final_winner, final_name, match_stats


def train_two_agents(
	agent1,        # SAC agent for player1
	agent2,        # SAC agent for player2
	env,           # Training environment (two-player)
	eval_env,      # Evaluation environment (two-player, same or different)
	path = "videos", # Path to save videos
	max_steps=2_000_000,
	start_steps=10_000,
	eval_interval=50_000,
	log_interval=1000,
	num_eval_episodes=200
):
	"""
	Jointly train two SAC agents in a two-player environment.

	Args:
		agent1 (SAC): First agent (player1).
		agent2 (SAC): Second agent (player2).
		env (gym.Env): A two-player environment returning separate rewards for each player.
		eval_env (gym.Env): Environment used for evaluation.
		max_steps (int): Max training steps (time steps).
		start_steps (int): Steps of random actions before learning.
		eval_interval (int): Steps between evaluations.
		num_eval_episodes (int): Number of episodes to evaluate each time.
	"""

	# log_file
	# leave the last part of the path
	log_path = path.split("/")[:-1]
	log_path = "/".join(log_path) + "/log.txt"
	
	
	# Initialize tracking variables
	current_step = 0
	# Reset environment
	obs1, info1 = env.reset()
	agent1.obs_normalizer.update(obs1)
	obs1 = agent1.obs_normalizer.normalize(obs1)
	
	# Agent2 perspective (mirrored observation, if needed)
	obs2 = env.obs_agent_two()
	agent2.obs_normalizer.update(obs2)	
	obs2 = agent2.obs_normalizer.normalize(obs2)
 
	done = False

	# We can track losses for each agent
	critic_loss_1, actor_loss_1, alpha_loss_1 = [], [], []
	critic_loss_2, actor_loss_2, alpha_loss_2 = [], [], []
	eval_rewards = []

	# Main training loop
	pbar = tqdm(range(max_steps), desc="Training", unit="step")
	
	# for 25 percent of the time we will use the alternative opponent 
	alternative_opponent = h_env.BasicOpponent(weak=False)
	
 
	while current_step < max_steps:
		# 1) Select actions
		# if current_step < start_steps:
		#     # Random actions before training
		#     action1 = np.random.uniform(-1, 1, size=agent1.action_dim)
		#     action2 = np.random.uniform(-1, 1, size=agent2.action_dim)
		# else:
		 
		rand_num = random.random()
		if rand_num < 0.1:
			action1 = agent1.select_action(obs1, deterministic=False)
			action2 = alternative_opponent.act(
				agent2.obs_normalizer.denormalize(obs2)
			)
		elif rand_num < 0.2:
			action1 = alternative_opponent.act(
				agent1.obs_normalizer.denormalize(obs1)
			)
			action2 = agent2.select_action(obs2, deterministic=False)
		else:
			action1 = agent1.select_action(obs1, deterministic=False)
			action2 = agent2.select_action(obs2, deterministic=False)
		
		
		# 2) Step environment with both actions
		#   The environment must return separate rewards for each agent OR a single reward if it's zero-sum.
		#   Suppose env.step() returns (next_obs1, reward1, reward2, done, info).
		#   If your environment only returns a single reward, you'll need to mirror or compute the second agent's reward.
		# env.render(mode="human")
		next_obs, r1, done, truncated, info = env.step(np.hstack([action1, action2]))

		# Next observation for agent2
		next_obs2 = env.obs_agent_two()
		
		agent1.obs_normalizer.update(next_obs)
		next_obs = agent1.obs_normalizer.normalize(next_obs)
		
		agent2.obs_normalizer.update(next_obs2)
		next_obs2 = agent2.obs_normalizer.normalize(next_obs2)
		
		
		info2 = env.get_info_agent_two()
		r2 = env.get_reward_agent_two(info2)
		
		# Normalize the rewards
		agent1.reward_normalizer.update(r1)
		r1 = agent1.reward_normalizer.normalize(r1)
		
		agent2.reward_normalizer.update(r2)
		r2 = agent2.reward_normalizer.normalize(r2)
		
		# 3) Store transitions in each agent's replay buffer
		#   Each agent has its own perspective of (state, action, reward, next_state).
		is_terminal = done or truncated
		agent1.replay_buffer.add(obs1, action1, r1, next_obs, float(is_terminal))
		agent2.replay_buffer.add(obs2, action2, r2, next_obs2, float(is_terminal))

		# 4) Move forward
		obs1 = next_obs
		obs2 = next_obs2
		current_step += 1
		pbar.update(1)

		# If done, reset environment
		if done or truncated:
			obs1, info1 = env.reset()
			obs2 = env.obs_agent_two()
			done = False

		# 5) One-step training updates for each agent
		#   We do multiple updates or just one. Example: one update per step.
		if current_step > start_steps:
			# Agent1 update
			loss1 = agent1.one_step_train(agent1.hyperparams["train"]["batch_size"])
			if loss1 is not None:
				critic_loss_1.append(loss1["Critic Loss"])
				actor_loss_1.append(loss1["Actor Loss"])
				alpha_loss_1.append(loss1["Alpha Loss"])

			# Agent2 update
			loss2 = agent2.one_step_train(agent2.hyperparams["train"]["batch_size"])
			if loss2 is not None:
				critic_loss_2.append(loss2["Critic Loss"])
				actor_loss_2.append(loss2["Actor Loss"])
				alpha_loss_2.append(loss2["Alpha Loss"])

			# Soft update target networks
			#   Agent1
			if current_step % agent1.hyperparams["train"]["target_update_interval"] == 0:
				agent1.update_target_network()
			#   Agent2
			if current_step % agent2.hyperparams["train"]["target_update_interval"] == 0:
				agent2.update_target_network()

		if current_step % log_interval == 0:
			if critic_loss_1 and critic_loss_2:
				pbar.set_postfix({
					"Critic Loss 1": sum(critic_loss_1) / len(critic_loss_1),
					"Actor Loss 1": sum(actor_loss_1) / len(actor_loss_1),
					"Alpha Loss 1": sum(alpha_loss_1) / len(alpha_loss_1),
					"Critic Loss 2": sum(critic_loss_2) / len(critic_loss_2),
					"Actor Loss 2": sum(actor_loss_2) / len(actor_loss_2),
					"Alpha Loss 2": sum(alpha_loss_2) / len(alpha_loss_2),
				})
				with open(log_path, "a") as f:
					f.write(f"Step {current_step}: Critic Loss 1 = {sum(critic_loss_1) / len(critic_loss_1):.2f}\n")
					f.write(f"Step {current_step}: Actor Loss 1 = {sum(actor_loss_1) / len(actor_loss_1):.2f}\n")
					f.write(f"Step {current_step}: Alpha Loss 1 = {sum(alpha_loss_1) / len(alpha_loss_1):.2f}\n")
					f.write(f"Step {current_step}: Critic Loss 2 = {sum(critic_loss_2) / len(critic_loss_2):.2f}\n")	
					f.write(f"Step {current_step}: Actor Loss 2 = {sum(actor_loss_2) / len(actor_loss_2):.2f}\n")
					f.write(f"Step {current_step}: Alpha Loss 2 = {sum(alpha_loss_2) / len(alpha_loss_2):.2f}\n")

		# 6) Evaluate periodically
		if current_step % eval_interval == 0:
			# def eval(self, env, player2, num_episodes,current_step, video_dir="videos", fps=30):
			avg_reward = agent1.eval(eval_env, agent2, num_eval_episodes, current_step, video_dir=path,log_path=log_path)
			eval_rewards.append(sum(avg_reward) / len(avg_reward))
			print(f"Step {current_step}: Eval Reward = {sum(avg_reward) / len(avg_reward):.2f}")
			with open(log_path, "a") as f:
				f.write(f"Step {current_step}: Eval Reward = {sum(avg_reward) / len(avg_reward):.2f}\n")
				
			# save both models 
			agent1_save = path + "/agent1/" + str(current_step)
			agent2_save = path + "/agent2/" + str(current_step)
			os.makedirs(agent1_save, exist_ok=True)
			os.makedirs(agent2_save, exist_ok=True)	
			
			agent1.save_model(agent1_save)
			agent2.save_model(agent2_save)
			
			

	pbar.close()
	return {
		"critic_loss_1": critic_loss_1,
		"actor_loss_1": actor_loss_1,
		"alpha_loss_1": alpha_loss_1,
		"critic_loss_2": critic_loss_2,
		"actor_loss_2": actor_loss_2,
		"alpha_loss_2": alpha_loss_2,
		"eval_rewards": eval_rewards,
	}
