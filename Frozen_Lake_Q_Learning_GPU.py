import gymnasium as gym
import cupy as cp
import random
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time
import sys

# Generate a random 10x10 Frozen Lake map
random_map = generate_random_map(size=10, p=0.8)

# Set up environment with the random map
env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=False)  # deterministic

# Parameters
num_episodes = 2000
max_steps = 100
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1  # exploration rate

# GPU Q-table
q_table_gpu = cp.zeros((env.observation_space.n, env.action_space.n), dtype=cp.float32)

# Timing variable
update_q_value_time = 0

# GPU version of Q-value update
def update_q_value_gpu(q_table_gpu, state, action, reward, new_state, learning_rate, discount_factor):
    global update_q_value_time
    start_time = time.time()

    old_value = q_table_gpu[state, action]
    next_max = cp.max(q_table_gpu[new_state])
    new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
    q_table_gpu[state, action] = new_value

    update_q_value_time += time.time() - start_time

# Training loop
def train_q_learning_gpu():
    for episode in range(num_episodes):
        state, _ = env.reset()
        for step in range(max_steps):
            # Epsilon-greedy action
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = int(cp.argmax(q_table_gpu[state]).get())  # Fetch from GPU to CPU

            new_state, reward, done, truncated, _ = env.step(action)

            update_q_value_gpu(q_table_gpu, state, action, reward, new_state, learning_rate, discount_factor)

            state = new_state
            if done or truncated:
                break

if __name__ == "__main__":
    if "-t" in sys.argv:
        num_runs = 10
        total_train_times = []
        total_update_q_value_times = []

        # global update_q_value_time
        for _ in range(num_runs):
            # Reset timing
            update_q_value_time = 0

            start_time = time.time()
            train_q_learning_gpu()
            total_train_times.append(time.time() - start_time)
            total_update_q_value_times.append(update_q_value_time)

        avg_train_time = sum(total_train_times) / num_runs
        avg_update_q_value_time = sum(total_update_q_value_times) / num_runs

        print(f"Average runtime for train_q_learning_gpu(): {avg_train_time:.4f} seconds over {num_runs} runs")
        print(f"Average runtime for update_q_value_gpu(): {avg_update_q_value_time:.4f} seconds over {num_runs} runs")
    else:
        train_q_learning_gpu()
