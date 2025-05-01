import gymnasium as gym
import cupy as cp
import numpy as np
import random
import time
import sys
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Generate a random 10x10 Frozen Lake map
random_map = generate_random_map(size=10, p=0.8)
env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=False)

# Parameters
num_episodes = 2000
max_steps = 100
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1

num_states = env.observation_space.n
num_actions = env.action_space.n

# Q-table on GPU
q_table_gpu = cp.zeros((num_states, num_actions), dtype=cp.float32)

# Timing
update_q_value_time = 0

# Experience buffers
state_buffer = []
action_buffer = []
reward_buffer = []
new_state_buffer = []

# Batch update function
def batch_update_q_values():
    global q_table_gpu, update_q_value_time
    if not state_buffer:
        return

    start_time = time.time()

    # Copy buffers to GPU arrays
    states = cp.array(state_buffer, dtype=cp.int32)
    actions = cp.array(action_buffer, dtype=cp.int32)
    rewards = cp.array(reward_buffer, dtype=cp.float32)
    new_states = cp.array(new_state_buffer, dtype=cp.int32)

    # Compute target values
    old_values = q_table_gpu[states, actions]
    next_max = cp.max(q_table_gpu[new_states], axis=1)
    targets = rewards + discount_factor * next_max
    new_values = (1 - learning_rate) * old_values + learning_rate * targets

    # Write back updated Q-values
    q_table_gpu[states, actions] = new_values

    update_q_value_time += time.time() - start_time

    # Clear the buffers
    state_buffer.clear()
    action_buffer.clear()
    reward_buffer.clear()
    new_state_buffer.clear()

# Training function
def train_q_learning_gpu_batched():
    for episode in range(num_episodes):
        state, _ = env.reset()
        for step in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = int(cp.argmax(q_table_gpu[state]).get())

            new_state, reward, done, truncated, _ = env.step(action)

            state_buffer.append(state)
            action_buffer.append(action)
            reward_buffer.append(reward)
            new_state_buffer.append(new_state)

            state = new_state
            if done or truncated:
                break

        batch_update_q_values()

if __name__ == "__main__":
    if "-t" in sys.argv:
        num_runs = 10
        total_train_times = []
        total_update_q_value_times = []

        for _ in range(num_runs):
            update_q_value_time = 0
            q_table_gpu = cp.zeros((num_states, num_actions), dtype=cp.float32)

            start_time = time.time()
            train_q_learning_gpu_batched()
            total_train_times.append(time.time() - start_time)
            total_update_q_value_times.append(update_q_value_time)

        avg_train_time = sum(total_train_times) / num_runs
        avg_update_q_value_time = sum(total_update_q_value_times) / num_runs

        print(f"Average runtime for train_q_learning_gpu_batched(): {avg_train_time:.4f} seconds over {num_runs} runs")
        print(f"Average runtime for batch_update_q_values(): {avg_update_q_value_time:.4f} seconds over {num_runs} runs")
    else:
        train_q_learning_gpu_batched()
