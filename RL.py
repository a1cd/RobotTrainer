import random
from typing import Union, List

import numpy as np
import tensorflow as tf
from numpy import ndarray


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, activation='relu'):
        """
        Initializes the DQN agent
        :param state_size: the number of values in the state
        :param action_size: the number of actions the agent can take
        :param learning_rate: the learning rate for the neural network (default 0.001) (between 0 and 1 but reasonable values range from 0.0001 to 0.001)
        :param discount_factor: the discount factor for the neural network (default 0.99) (between 0 and 1 but reasonable values range from 0.9 to 0.99)
        :param epsilon: the initial epsilon value for the epsilon greedy algorithm (default 1.0) (between 0 and 1 but reasonable values range from 0.9 to 1.0)
        :param epsilon_decay: the decay rate for epsilon (default 0.995) (between 0 and 1 but reasonable values range from 0.9 to 1.0)
        :param epsilon_min: the minimum value for epsilon (default 0.01) (between 0 and 1 but reasonable values range from 0.01 to 0.1)
        :param activation: the activation function for the neural network (default 'relu') (options: 'relu', 'sigmoid', 'tanh' or 'linear')
            - relu is the most common activation function, usefully for hidden layers
            - sigmoid is useful for output layers when the output is a probability
            - tanh is useful for output layers when the output is a probability
            - linear is useful for output layers when the output is a value
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = []
        self.activation: str = activation
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(int(self.state_size * .75), input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(int(self.state_size * .75), activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state: ndarray[Union[int, float]], action: Union[int, ndarray[int]], reward: float, next_state: ndarray[Union[int, float]], done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: ndarray[Union[int, float]]) -> List[float]:
        if np.random.rand() <= self.epsilon:
            return [random.uniform(-1, 1) for _ in range(self.action_size)]
        q_values = self.model.predict(state)
        normalized_q_values = 2 * (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values)) - 1
        return normalized_q_values[0].tolist()

    # def replay(self, batch_size):
    #     if len(self.memory) < batch_size:
    #         return
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             q_next = np.max(self.model.predict(next_state)[0])
    #             target = (reward + self.discount_factor * q_next)
    #         q_values: ndarray[Union[int, float]] = self.model.predict(state)
    #         # q_values[0][action] = target
    #         q_values[0][action[0]] = target
    #         self.model.fit(state, q_values, epochs=1, verbose=0) # <- the probatic line
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                q_next = np.max(self.model.predict(next_state)[0])
                target = (reward + self.discount_factor * q_next)
            q_values: ndarray[Union[int, float]] = self.model.predict(state)
            action_index = np.argmax(action)
            q_values[0][action_index] = target
            self.model.fit(state, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def reset(self):
        self.memory = []
        self.model = self._build_model()
        self.epsilon = 1.0

    def get_state(self):
        return self.model.get_weights()