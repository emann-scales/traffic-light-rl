import traci
import numpy as np
import random

# Q-Learning Parameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

num_states = 100
num_actions = 2  # {0: keep phase, 1: switch phase}
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

def main():
    print("Starting Traffic Light RL Simulation (Demo)")
    for episode in range(5):  # demo episodes
        state = random.randint(0, num_states - 1)
        action = choose_action(state)
        reward = random.randint(-10, 10)  # placeholder reward
        next_state = random.randint(0, num_states - 1)
        update_q(state, action, reward, next_state)
        print(f"Episode {episode+1}: State={state}, Action={action}, Reward={reward}")

if __name__ == "__main__":
    main()
