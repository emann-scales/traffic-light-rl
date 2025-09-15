import traci
import numpy as np
import random

# Q-Learning Parameters
alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
epsilon = 1.0   # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

# Initialize Q-table
num_states = 100  # simplified representation
num_actions = 2   # {0: keep phase, 1: switch phase}
Q = np.zeros((num_states, num_actions))

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1])  # explore
    return np.argmax(Q[state])  # exploit

def update_q(state, action, reward, next_state):
    best_next_action = np.argmax(Q[next_state])
    Q[state, action] = Q[state, action] + alpha * (
        reward + gamma * Q[next_state, best_next_action] - Q[state, action]
    )
