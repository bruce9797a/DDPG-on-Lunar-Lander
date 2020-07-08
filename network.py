import torch
import torch.nn as nn
import torch.nn.functional as F

class policy_network(nn.Module):
    
    def __init__(self , num_state , num_action):
        
        super().__init__()
        self.fc1 = nn.Linear(num_state , 128 )
        self.ln1 = nn.LayerNorm(128)
        ## more hidden
        self.fc2 = nn.Linear(128 , 64 )
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64,32)
        self.ln3 = nn.LayerNorm(32)
        self.out = nn.Linear(32 , num_action )
        
    def forward(self ,x):
        x = F.relu(self.fc1(x))
        x = self.ln1(x)
        x = F.relu(self.fc2(x))
        x = self.ln2(x)
        x = F.relu(self.fc3(x))
        x = self.ln3(x)
        x = torch.tanh(self.out(x))
        return x
    
class value_network(nn.Module):
    
    def __init__(self ,num_state,num_action):
        
        super().__init__()
        self.fc1 = nn.Linear(num_state, 128 )
        self.ln1 = nn.LayerNorm(128)
        ## more hidden
        self.fc2 = nn.Linear(128+num_action , 64 )
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64,32)
        self.ln3 = nn.LayerNorm(32)
        self.out = nn.Linear(32 , num_action)
        
    def forward(self,s,a):
        x = F.relu(self.fc1(s))
        x = self.ln1(x)
        x = torch.cat([x,a],1)
        x = F.relu(self.fc2(x))
        x = self.ln2(x)
        x = F.relu(self.fc3(x))
        x = self.ln3(x)
        x = self.out(x)
        return x




