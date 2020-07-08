import torch
from network import policy_network,value_network
from buffer import  ReplayBuffer
import numpy as np


class DDPG():
    
    def __init__(self , num_state , num_action):
        self.actor = policy_network(num_state , num_action)
        self.critic = value_network(num_state,num_action)
        
        self.target_a =  policy_network(num_state , num_action)
        self.target_a.load_state_dict(self.actor.state_dict())
        
        self.target_q = value_network(num_state,num_action)
        self.target_q.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters()   , lr=0.001,amsgrad=True)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters()  , lr=0.001,amsgrad=True)
        
        self.num_state = num_state
        self.num_action = num_action
        
        self.batch_size = 64
        self.buffer = ReplayBuffer( 4096 )
        
        
    def take_action(self , x  ) :
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        action = self.actor(x)
        return action
            
    
    def store_transition(self, state , action , reward , next_state , done ):
        
        self.buffer.add(state , action , reward , next_state , done)
    
    def update_parameters(self):
        
        if len(self.buffer) < self.batch_size:
            return
        
        gamma = 0.99
        tau = 0.001
        
        loss_fn = torch.nn.MSELoss(reduction = 'mean')
        
        batch = self.buffer.sample(self.batch_size)
        states , actions , rewards , next_states , dones = batch
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards).view(-1,1)
        next_states = torch.from_numpy(next_states)
        
        #using target_actor network to predict action
        next_actions = self.target_a(next_states)
        next_Q = self.target_q(next_states,next_actions)
        target_values = rewards + gamma*next_Q
        
        predict_Q = self.critic(states,actions)
        
        critic_loss = loss_fn(target_values,predict_Q)
        #update critic using loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = - self.critic(states,self.actor(states))
        actor_loss = actor_loss.mean()
        #update actor using loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #soft replace
        for target_param, param in zip(self.target_q.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_a.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            




