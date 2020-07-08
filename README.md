# DDPG-on-Lunar-Lander

# DDPG
pytorch implementation of Deep deterministic policy gradient

# About Lunar-Lander-v2(Continuous)
State space  
state[0] : horizontal coordinate  
state[1] : vertical coordinate  
state[2] : horizontal speed  
state[3] : vertical speed  
state[4] : angle  
state[5] : angular speed  
state[6] : 1 if first leg has contact, else 0  
state[7] : 1 if second leg has contact, else 0  
Action space  
action[0] : main engine , -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.  
action[1] : Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off  

Reward  
In each time step (an action taken):  
-0.3, if fire the main engine  
-0.03, if fire side engine  
  
-100, if crashed  
-100, fly outside the given range (Position X ≥ 1.0)  
  
+100, if landing successfully   
+10, each leg contact  
  
# Run  
training:  
python train.py  
![image](https://github.com/bruce9797a/DDPG-on-Lunar-Lander/blob/master/reward_history.JPG)  
testing:  
python test.py  
![image](https://github.com/bruce9797a/DDPG-on-Lunar-Lander/blob/master/result.gif)
