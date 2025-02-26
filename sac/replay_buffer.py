import torch 
import numpy as np
from numba import jit
import numpy as np



class ReplayBuffer:
    """
    A simple FIFO replay buffer for SAC agents.
    It can get multiple environments and store the transitions in a single buffer.
    """
    
    def __init__(self,obs_shape, act_shape, max_size, buffer_device='cpu',num_envs=1):
        
        self.max_size = max_size # Maximum size of the buffer
        self.current_size = 0 # Current size of the buffer
        self.ptr = 0 # Pointer to the current position in the buffer
        self.num_envs = num_envs
        
        self.states = torch.zeros((self.max_size, obs_shape), dtype=torch.float32).to(buffer_device)
        self.actions = torch.zeros((self.max_size, act_shape), dtype=torch.float32).to(buffer_device)
        self.rewards = torch.zeros((self.max_size, 1), dtype=torch.float32).to(buffer_device)
        self.next_states = torch.zeros((self.max_size, obs_shape), dtype=torch.float32).to(buffer_device)
        self.dones = torch.zeros((self.max_size, 1), dtype=torch.float32).to(buffer_device)
                
        self.buffer_device = buffer_device # if you have enough memory, you can hold all the data in the GPU
        
        
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        takes np arrays as input
        and stores them in the buffer as torch tensors.
        Args:
            state (np.array): The current state (num_envs, obs_shape)
            action (np.array): The action taken (num_envs, act_shape)
            reward (float): The reward received (num_envs,reward_shape)
            next_state (np.array): The next state (num_envs, obs_shape)
            done (bool): Whether the episode is done (num_envs,1)
        """

        state, action, reward, next_state, done = self.convert_to_tensor(state), \
                                                self.convert_to_tensor(action), \
                                                self.convert_to_tensor([reward]), \
                                                self.convert_to_tensor(next_state), \
                                                self.convert_to_tensor([done]) 
                                                
        # Add the transition to the buffer using the current pointer
    
        if self.ptr + self.num_envs > self.max_size:
            self.ptr = self.ptr - self.max_size
        
        self.states[self.ptr:self.ptr+self.num_envs] = state
        self.actions[self.ptr:self.ptr+self.num_envs] = action
        self.rewards[self.ptr:self.ptr+self.num_envs] = reward
        self.next_states[self.ptr:self.ptr+self.num_envs] = next_state
        self.dones[self.ptr:self.ptr+self.num_envs] = done
        
        self.ptr += self.num_envs
        self.current_size = min(self.current_size + self.num_envs, self.max_size)
        
        
    def convert_to_tensor(self, data):
        """
        Convert a numpy array to a torch tensor.
        """
        return torch.tensor(data).to(self.buffer_device)
    
    def sample(self,batch_size, device='cpu'):
        """
        Sample a batch of transitions from the buffer.
        """
        idxs = np.random.randint(0, self.current_size, size=batch_size)
        
        batch = (self.states[idxs].to(device),
                 self.actions[idxs].to(device),
                 self.rewards[idxs].to(device),
                 self.next_states[idxs].to(device),
                 self.dones[idxs].to(device))
        
        return batch
    

    def __len__(self):
        return self.current_size
    
    def clear(self):
        self.current_size = 0
        self.ptr = 0
        
    def save(self, path):
        
        state_dict = {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_states': self.next_states,
            'dones': self.dones,
            'ptr': self.ptr,
            'current_size': self.current_size
        }
        
        torch.save(state_dict, path)
        
    def load(self, path):
        state_dict = torch.load(path)
        
        self.states = state_dict['states']
        self.actions = state_dict['actions']
        self.rewards = state_dict['rewards']
        self.next_states = state_dict['next_states']
        self.dones = state_dict['dones']
        self.ptr = state_dict['ptr']
        self.current_size = state_dict['current_size']
        
        return self


    
    
        
