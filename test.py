import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from ddpg import DDPG
from noise import OUNoise
import torch

env = gym.make('LunarLanderContinuous-v2').unwrapped

ac_low = env.action_space.low
ac_high = env.action_space.high
num_state = env.observation_space.shape[0]
num_action = env.action_space.shape[0]
#train
agent = torch.load("LunarLander_agent.pkl")
reward_history = []
noise = OUNoise(env.action_space,theta=0.15, max_sigma=0.2, min_sigma=0.2)

for e in tqdm(range(1,101)):
    state = env.reset()
    done = False
    reward_sum = 0.0
    step = 0
    while done == False :
        env.render()
        action = agent.take_action(state)
        action = noise.get_action(action.detach().numpy(),step)
        next_state , reward , done , _ = env.step(action)
        reward_sum += reward
        #reward-shaping
        reward = reward -abs(next_state[0]) + 1/(abs(next_state[1])+0.1)-abs(next_state[4])
        agent.store_transition( state , action , reward , next_state , done )
        state = next_state[:]
        step += 1
    reward_history.append(reward_sum)        

env.close()

plt.plot(reward_history)
plt.savefig("test_reward_history")
