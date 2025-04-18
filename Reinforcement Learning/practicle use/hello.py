import gymnasium as gym
import numpy as np
# import gym
cliffENV = gym.make('CliffWalking-v0')

# a boolean value tell weather the episode is finished or not
done = False

state = cliffENV.reset()

while not done:
    action = int(np.random.randint(low=0, high=4, size=1))
    state, reward, done, _ = cliffENV.step(action)

cliffENV.close()
