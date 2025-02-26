import torch 
from torch import nn
import torch.nn.functional as F


class SacCritic(nn.Module):
    '''
    Implement the Q network in the SAC algorithm.
    We will use double Q networks. and we will use the one with the minimum value.
    '''
    
    def __init__(self,model_kwargs,state_dim, action_dim):
        '''
        Initialize the Q network.
        
        model_kwargs: dict
            A dictionary containing the hyperparameters of the model.
            model_kwargs has hidden size and number of hidden layers.
        '''
        super(SacCritic, self).__init__()
        
        hidden_size = model_kwargs.get('hidden_size', 256)
        num_hidden_layers = model_kwargs.get('num_hidden_layers', 2)
        
        layers1 = [nn.Linear(state_dim + action_dim, hidden_size), nn.ELU()]
        for _ in range(num_hidden_layers - 1):
            layers1.extend([nn.Linear(hidden_size, hidden_size), nn.ELU()])
        layers1.append(nn.Linear(hidden_size, 1))
        self.net1 = nn.Sequential(*layers1)  

        # Define Q-network 2
        layers2 = [nn.Linear(state_dim + action_dim, hidden_size), nn.ELU()]
        for _ in range(num_hidden_layers - 1):
            layers2.extend([nn.Linear(hidden_size, hidden_size), nn.ELU()])
        layers2.append(nn.Linear(hidden_size, 1))
        self.net2 = nn.Sequential(*layers2)  
        
        
        
    def forward(self, state, action):
        '''
        Forward pass of the Q network.
        Concatenate the state and action and pass it through the network.
        '''
        
        x = torch.cat([state, action], dim=1)
        
        q1 = self.net1(x)
        q2 = self.net2(x)
        
        return q1, q2
    
                
        
        
        
    
