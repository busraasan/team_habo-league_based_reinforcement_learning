import numpy as np
import torch
import hockey.hockey_env as h_env
from hockey.hockey_env import HockeyEnv, BasicOpponent
from tqdm import tqdm
import os 
import cv2
from matplotlib import pyplot as plt
from TD3.utils import running_mean
from sac.sac import SAC
from TD3.td3 import TD3Agent
import random

def plot_stats(arr, title, y_label, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(arr, label=title, alpha=0.1, color="blue")
    plt.plot(running_mean(arr,20), label="smoothed-"+title, color="blue")
    plt.xlabel("Steps")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def add_noise(model, noise_std):
	for param in model.actor.parameters():
		if random.random() < 0.2:
			param.data += torch.normal(0, noise_std, param.data.shape).to(param.data.device)
	for param in model.critic.parameters():
		if random.random() < 0.2:
			param.data += torch.normal(0, noise_std, param.data.shape).to(param.data.device)
   
	return model

class TD3Trainer:
    def __init__(self, config, logger=None):
        """
        Args:
            config (dict): Configuration dictionary. 
                           Should contain keys like 'opponent' to choose between 'weak' or 'strong'.
            logger: Optional logger for experiment tracking.
        """
        self.logger = logger
        self._config = config

        self.state_dim = self._config["state_dim"]
        self.action_dim = self._config["action_dim"]
        self.max_action = self._config["max_action"]
        self.device = self._config["device"]
        self.num_envs = self._config["num_envs"]

        self.strong_opponent = h_env.BasicOpponent(weak=False)

        self.weak_opponent = h_env.BasicOpponent(weak=True)

        self.td3_opponent = TD3Agent(state_dim=self.state_dim, 
                                action_dim=self.action_dim, 
                                max_action=self.max_action)
        
        self.sac_opponent = SAC(state_dim=self.state_dim, 
                                action_dim=self.action_dim, 
                                max_action=self.max_action, 
                                device=self.device, 
                                num_envs=self.num_envs,
                                hyperparam_yaml="../sac/hyperparams.yaml")
        
        # load trained weights to sac and td3
        self.td3_opponent.load_weights(path_actor="../TD3/strong_agent_results/weights/td3_vs_strong_opponent_model/actor.pth", path_critic="../TD3/strong_agent_results/weights/td3_vs_strong_opponent_model/critic.pth")
        self.sac_opponent.load_model("../sac/weights/agent_2")

        # pick opponent
        if self._config["opponent"] == "strong":
            self.opponent = self.strong_opponent
        elif self._config["opponent"] == "weak":
            self.opponent = self.weak_opponent
        elif self._config["opponent"] == "SAC":
            self.opponent = self.sac_opponent
        elif self._config["opponent"] == "TD3":
            self.opponent = self.td3_opponent
        else:
            raise ValueError(f"Unknown opponent type: {self._config['opponent']}")            


    def train_td3(self, TD3_agent, env, max_steps=500, eval_step=500, path = "results", save_step = 1000000):
        """
        Train a TD3 agent in the HockeyEnv, similarly to the DQN training loop.

        Args:
            TD3_agent: Your TD3 agent object (must have .act() and .update() methods).
            env:       The environment (HockeyEnv or similar).
            max_steps: Number of environment steps to run training.
            eval_step: Evaluate the agent every 'eval_step' environment steps.

        Returns:
            (losses, win_rates): Two lists tracking loss and win rate over time.
        """

        actor_losses = []
        critic_losses = []
        win_rates = []
        win_rates_sac = []

        overall_step = 0
        done = False

        if self._config["continue_training"]:
            TD3_agent.load_weights(path_actor="../TD3/strong_agent_results/weights/td3_vs_strong_opponent_model/actor.pth", path_critic="../TD3/strong_agent_results/weights/td3_vs_strong_opponent_model/critic.pth")

        obs, info = env.reset()

        print("Filling buffer with random (or pre-training) actions...")
        initial_fill_steps = 600
        for _ in range(initial_fill_steps):
            done = False
            while not done:
                action_agent = np.random.uniform(low=-1.0, high=1.0, size=(4,))
                obs_opponent = env.obs_agent_two()
                action_opponent = self.opponent.act(obs_opponent)
                obs_new, reward, done, trunc, info = env.step(np.hstack([action_agent, action_opponent]))

                # Reward normalization if desired
                # self.reward_normalizer.update(reward)
                # reward_norm = self.reward_normalizer.normalize(reward)

                # Store in replay buffer
                TD3_agent.store_transition(obs, action_agent, reward, obs_new, done)

                obs = obs_new

            obs, info = env.reset()
        print("Initial buffer fill complete.")

        # Training Loop
        done = False
        obs, info = env.reset()
        pbar = tqdm(range(max_steps), desc="Training TD3")

        for step_i in range(max_steps):

            if done:
                obs, info = env.reset()

            if self._config["selfplay"]:

                if self.opponent is None or isinstance(self.opponent, BasicOpponent) and step_i % 40000 < 20000:
                    self.opponent = self.td3_opponent

                elif isinstance(self.opponent, TD3Agent) and step_i % 40000 > 20000:
                    self.opponent = BasicOpponent(weak=False)

                if step_i % 100000 == 0:
                    self.td3_opponent.actor.load_state_dict(TD3_agent.actor.state_dict())
                    self.td3_opponent.critic.load_state_dict(TD3_agent.critic.state_dict())
                
            # 1) Agent acts (continuous)
            action_agent = TD3_agent.act(obs, add_noise=True)  # Possibly add exploration noise
            # 2) Opponent acts
            obs_opponent = env.obs_agent_two()
            action_opponent = self.opponent.act(obs_opponent)

            # 3) Environment step
            obs_new, reward, done, trunc, info = env.step(np.hstack([action_agent, action_opponent]))

            # Normalize reward if desired
            # self.reward_normalizer.update(reward)
            # reward_norm = self.reward_normalizer.normalize(reward)

            TD3_agent.store_transition(obs, action_agent, reward, obs_new, done)
            update_info = TD3_agent.update()  

            if isinstance(update_info, dict):
                actor_losses.append(update_info["actor_loss"])
                critic_losses.append(update_info["critic_loss"])
                
            obs = obs_new
            overall_step += 1

            # Print progress every 1000 steps
            pbar.update(1)
            if (overall_step % 1000) == 0:
                mean_critic_loss = np.mean(critic_losses[-100:]) if len(critic_losses) > 100 else np.mean(critic_losses)
                mean_actor_loss = np.mean(actor_losses[-100:]) if len(actor_losses) > 100 else np.mean(actor_losses)
                pbar.set_postfix({
                    "Actor Loss": f"{mean_actor_loss:.3f}",
                    "Critic Loss": f"{mean_critic_loss:.3f}"
                })

                with open(path+"/losses.txt", "a") as f:
                    f.write(f"{overall_step}, {mean_actor_loss}, {mean_critic_loss}\n")
                
            # Evaluate periodically
            if ((overall_step % eval_step) == 0) and (overall_step > 0):
                
                win_rate = self.evaluate_td3(TD3_agent, opponent="strong", current_step=step_i,video_dir= os.path.join(path, "videos_strong") ,max_episodes=100)
                print(f"Evaluation STRONG at step={overall_step}, win rate={win_rate[0]:.2f}%")

                win_rate_sac = self.evaluate_td3(TD3_agent, opponent="SAC", current_step=step_i,video_dir= os.path.join(path, "videos_sac") ,max_episodes=100)
                print(f"Evaluation SAC at step={overall_step}, win rate={win_rate_sac[0]:.2f}%")
                
                win_rates.append(win_rate)
                with open(path+"/win_rates.txt", "a") as f:
                    f.write(f"{overall_step},{win_rate}\n")

                win_rates_sac.append(win_rate_sac)
                with open(path+"/win_rates_sac.txt", "a") as f:
                    f.write(f"{overall_step},{win_rate_sac}\n")

                plot_stats(actor_losses, "Actor Loss", "Loss", path+"/actor_loss.png")
                plot_stats(critic_losses, "Critic Loss", "Loss", path+"/critic_loss.png")
                plot_stats([x[0] for x in win_rates], "Win Rate", "Win Rate", path+"/win_rate.png")
                plot_stats([x[0] for x in win_rates_sac], "Win Rate", "Win Rate", path+"/win_rate_sac.png")

            if ((overall_step % save_step) == 0) and (overall_step > 0):
                os.makedirs(os.path.join(path, "weights_"+str(overall_step)), exist_ok=True)
                TD3_agent.save_weights(os.path.join(path, "weights_"+str(overall_step)))

        return critic_losses, actor_losses, win_rates, win_rates_sac
    
    def train_td3_selfplay(self, TD3_agent, env, max_steps=500, eval_step=500, path = "results", save_step = 1000000, add_sac=False):
        
        actor_losses = []
        critic_losses = []
        win_rates = []
        win_rates_sac = []

        overall_step = 0
        done = False

        if self._config["continue_training"]:
            TD3_agent.load_weights(path_actor="../TD3/strong_agent_results/weights/td3_vs_strong_opponent_model/actor.pth", path_critic="../TD3/strong_agent_results/weights/td3_vs_strong_opponent_model/critic.pth")

        obs, info = env.reset()

        # Training Loop
        done = False
        obs, info = env.reset()
        pbar = tqdm(range(max_steps), desc="Training TD3")

        for step_i in range(max_steps):

            if done:
                obs, info = env.reset()

                rand_num = random.random()

                if add_sac:
                    if rand_num < 0.15:
                        self.opponent = self.strong_opponent
                    elif rand_num < 0.15:
                        self.opponent = self.sac_opponent
                    else:
                        self.opponent = self.td3_opponent
                else:
                    if rand_num < 0.2:
                        self.opponent = self.strong_opponent
                    else:
                        self.opponent = self.td3_opponent


            if step_i % 100000 == 0:
                self.td3_opponent.actor.load_state_dict(TD3_agent.actor.state_dict())
                self.td3_opponent.critic.load_state_dict(TD3_agent.critic.state_dict())

            action_agent = TD3_agent.act(obs, add_noise=True)  # Possibly add exploration noise
            obs_opponent = env.obs_agent_two()
            action_opponent = self.opponent.act(obs_opponent)

            obs_new, reward, done, trunc, info = env.step(np.hstack([action_agent, action_opponent]))

            TD3_agent.store_transition(obs, action_agent, reward, obs_new, done)
            update_info = TD3_agent.update()  

            if isinstance(update_info, dict):
                actor_losses.append(update_info["actor_loss"])
                critic_losses.append(update_info["critic_loss"])
                
            obs = obs_new
            overall_step += 1

            # Print progress every 1000 steps
            pbar.update(1)
            if (overall_step % 1000) == 0:
                mean_critic_loss = np.mean(critic_losses[-100:]) if len(critic_losses) > 100 else np.mean(critic_losses)
                mean_actor_loss = np.mean(actor_losses[-100:]) if len(actor_losses) > 100 else np.mean(actor_losses)
                pbar.set_postfix({
                    "Actor Loss": f"{mean_actor_loss:.3f}",
                    "Critic Loss": f"{mean_critic_loss:.3f}"
                })

                with open(path+"/losses.txt", "a") as f:
                    f.write(f"{overall_step}, {mean_actor_loss}, {mean_critic_loss}\n")
                
            # Evaluate periodically
            if ((overall_step % eval_step) == 0) and (overall_step > 0):
                
                win_rate = self.evaluate_td3(TD3_agent, opponent="strong", current_step=step_i,video_dir= os.path.join(path, "videos_strong") ,max_episodes=100)
                print(f"Evaluation STRONG at step={overall_step}, win rate={win_rate[0]:.2f}%")

                win_rate_sac = self.evaluate_td3(TD3_agent, opponent="SAC", current_step=step_i,video_dir= os.path.join(path, "videos_sac") ,max_episodes=100)
                print(f"Evaluation SAC at step={overall_step}, win rate={win_rate_sac[0]:.2f}%")
                
                win_rates.append(win_rate)
                with open(path+"/win_rates.txt", "a") as f:
                    f.write(f"{overall_step},{win_rate}\n")

                win_rates_sac.append(win_rate_sac)
                with open(path+"/win_rates_sac.txt", "a") as f:
                    f.write(f"{overall_step},{win_rate_sac}\n")

                plot_stats(actor_losses, "Actor Loss", "Loss", path+"/actor_loss.png")
                plot_stats(critic_losses, "Critic Loss", "Loss", path+"/critic_loss.png")
                plot_stats([x[0] for x in win_rates], "Win Rate", "Win Rate", path+"/win_rate.png")
                plot_stats([x[0] for x in win_rates_sac], "Win Rate", "Win Rate", path+"/win_rate_sac.png")

            if ((overall_step % save_step) == 0) and (overall_step > 0):
                os.makedirs(os.path.join(path, "weights_"+str(overall_step)), exist_ok=True)
                TD3_agent.save_weights(os.path.join(path, "weights_"+str(overall_step)))

        print("Filling buffer with random (or pre-training) actions...")
        initial_fill_steps = 600
        for _ in range(initial_fill_steps):
            done = False
            while not done:
                action_agent = np.random.uniform(low=-1.0, high=1.0, size=(4,))
                obs_opponent = env.obs_agent_two()
                action_opponent = self.opponent.act(obs_opponent)
                obs_new, reward, done, trunc, info = env.step(np.hstack([action_agent, action_opponent]))
                print(obs_new, reward, done, trunc, info)
                TD3_agent.store_transition(obs, action_agent, reward, obs_new, done)

                obs = obs_new

            obs, info = env.reset()
        print("Initial buffer fill complete.")

    def evaluate_td3(self, TD3_agent, current_step, video_dir, max_episodes=100, max_steps=500, opponent="strong"):
        """
        Evaluate TD3 agent against a chosen opponent.

        Args:
            TD3_agent: The trained TD3 agent.
            max_episodes (int): Number of episodes to run evaluation.
            max_steps (int): Max steps per episode before termination.

        Returns:
            [win%, loss%, tie%] as a list of floats.
        """

        # Create a fresh environment for evaluation
        eval_env = HockeyEnv()

        agent_won, bot_won, tie = 0, 0, 0

        for episode in range(max_episodes):
            obs, info = eval_env.reset()
            
            done = False
            frames = []

            for t in range(max_steps):
                
                if episode < 5:
                    frame = eval_env.render(mode = "rgb_array")
                if frame is not None:
                    frames.append(frame)

                # Agent chooses deterministic (or low-noise) action during evaluation
                action_agent = TD3_agent.act(obs, add_noise=False)
                obs_opponent = eval_env.obs_agent_two()

                if opponent == "strong":
                    self.opponent = self.strong_opponent
                elif opponent == "weak":
                    self.opponent = self.weak_opponent
                elif opponent == "SAC":
                    self.opponent = self.sac_opponent
                elif opponent == "TD3":
                    self.opponent = self.td3_opponent

                action_opponent = self.opponent.act(obs_opponent)

                # Step env
                obs, reward, done, trunc, info = eval_env.step(np.hstack([action_agent, action_opponent]))

                if done:
                    break
                
            if episode < 5:
                save_video(frames, os.path.join(video_dir, f"step_{current_step}_episode_{episode}.mp4"))
                
            # After the episode, check outcome
            if info["winner"] == 1:
                agent_won += 1
            elif info["winner"] == -1:
                bot_won += 1
            else:
                tie += 1

        total_eps = max_episodes
        agent_win_rate = (agent_won / total_eps) * 100
        bot_win_rate = (bot_won / total_eps) * 100
        tie_rate = (tie / total_eps) * 100

        print(f"[TD3 Evaluation] Over {max_episodes} episodes:")
        print(f" - Agent wins  : {agent_win_rate:.2f}%")
        print(f" - Opponent wins: {bot_win_rate:.2f}%")
        print(f" - Ties        : {tie_rate:.2f}%")

        return [agent_win_rate, bot_win_rate, tie_rate]


def save_video(frames, filename, fps=30):
	""" Saves a list of frames as a video using OpenCV. """
	height, width, _ = frames[0].shape
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use MP4 format
	out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

	for frame in frames:
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (OpenCV format)
		out.write(frame)

	out.release()