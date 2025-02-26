import os
import numpy as np

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
          
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
