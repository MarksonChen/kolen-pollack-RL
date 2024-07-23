import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from environments.gridworld import GridWorldEnv

# Define the neural network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
target_update = 10
memory_size = 10000
num_episodes = 500

# Environment
env = GridWorldEnv(render_mode=None)
state_dim = np.prod(env.observation_space['agent'].shape) * 2  # Agent and Target
action_dim = env.action_space.n

# Initialize the neural network and optimizer
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

def preprocess_state(state):
    agent_state = state['agent']
    target_state = state['target']
    return np.concatenate([agent_state, target_state])

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return policy_net(state).argmax().item()

def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.LongTensor(action_batch).unsqueeze(1)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for episode in range(num_episodes):
    state, _ = env.reset()
    state = preprocess_state(state)
    total_reward = 0

    for t in range(100):
        action = select_action(state, epsilon)
        next_state, reward, terminated, _, _ = env.step(action)
        next_state = preprocess_state(next_state)
        memory.append((state, action, reward, next_state, terminated))

        state = next_state
        total_reward += reward

        optimize_model()

        if terminated:
            break

    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")

torch.save(policy_net.state_dict(), os.path.join("output", "dqn_gridworld.pth"))
env.close()

loaded_model = DQN(state_dim, action_dim)
loaded_model.load_state_dict(torch.load(os.path.join("output", "dqn_gridworld.pth")))

env = GridWorldEnv(render_mode="human")

state, _ = env.reset()
state = preprocess_state(state)
for _ in range(100):
    action = select_action(state, epsilon_min)  # Use minimum epsilon for testing
    next_state, reward, terminated, _, _ = env.step(action)
    state = preprocess_state(next_state)
    env.render()
    if terminated:
        state, _ = env.reset()
        state = preprocess_state(state)

env.close()
