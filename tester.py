import numpy as np
import gym

from RL import DQNAgent

# Define the environment
env = gym.make('Navigation-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define the agent
agent = DQNAgent(state_size, action_size)

# Train the agent
num_episodes = 1000
batch_size = 32
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0
    while not done:
        # Take an action
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward

        # Remember the experience
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        # Update the agent
        agent.replay(batch_size)

    # Print the total reward for the episode
    print('Episode {}: Total reward = {}'.format(episode, total_reward))

# Save the trained agent
agent.save('agent.h5')
