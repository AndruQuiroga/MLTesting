import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os


env = gym.make("CartPole-v0")
state_size = 4
action_size = 2
batch_size = 32
n_episodes = 1000


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        self.gammma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        self.learning_rate = 0.001

        self.model = self.build_model()

    def build_model(self):

        model = Sequential()

        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = (reward + self.gammma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':
    agent = DQNAgent(state_size, action_size)
    done = False
    for e in range(n_episodes):

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(5000):

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print(f"ep: {e}/{n_episodes}, score: {time}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
