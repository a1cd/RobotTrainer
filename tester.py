from random import random

import numpy as np
from RL import DQNAgent
# check if rl learns to divide by 2 (very simple task)
agent = DQNAgent(1, 1, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.01, learning_rate=0.01, discount_factor=0.99)
for i in range(10000):
    # get random state, this represents the number to divide by 2, our input to the calculator
    state = np.array([random() * 100])
    # get the action, this represents what the neural network thinks the result should be
    action = agent.act(state)
    # get the reward, this represents how close the neural network's guess was to the actual result
    reward = 1 - abs(action[0] - state[0] / 2)
    # get the next state, this represents what the number to divide by 2 would be if the neural network's guess was correct
    next_state = np.array([action[0] * 2])
    # get the done value, this represents whether the neural network's guess was correct
    done = reward > 0.99
    # remember the state, action, reward, next state and done value
    agent.remember(state, action, reward, next_state, done)

    # if i is one, then create random memory
    if i == 1:
        for _ in range(100):
            # get random state, this represents the number to divide by 2, our input to the calculator
            state = np.array([random() * 100])
            # this will be random so the net can explore
            action = np.array([random() * 100])
            # get the reward, this represents how close the neural network's guess was to the actual result
            reward = 1 - abs(action[0] - state[0] / 2)
            # get the next state, this represents what the number to divide by 2 would be if the neural network's guess was correct
            next_state = np.array([action[0] * 2])
            # get the done value, this represents whether the neural network's guess was correct
            done = reward > 0.99
            # remember the state, action, reward, next state and done value
            agent.remember(state, action, reward, next_state, done)
    # train the neural network
    agent.replay(32)
    # print the neural network's guess and the actual result
    print(f"guess: {action[0]}, actual: {state[0] / 2}")