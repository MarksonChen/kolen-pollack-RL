import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from gridworld import GridWorldEnv
from bio_linear import FALinear
from kolen_pollack import KolenPollackMLP
from feedback_alignment import FeedbackAlignmentMLP
import os
import matplotlib.pyplot as plt

# Hyperparameters
gamma = 0.99
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
target_update = 10
memory_size = 10000
num_episodes = 600
hidden_sizes = [128]

env = GridWorldEnv(render_mode=None)
state_dim = np.prod(env.observation_space['agent'].shape) * 2  # Agent and Target
action_dim = env.action_space.n

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super(MLP, self).__init__()
        layers = []

        # Add hidden layers with ReLU activations
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        # Add the final layer without activation
        layers.append(nn.Linear(input_dim, output_dim))

        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

linear_types = ['fa', 'kp', 'bp'][1:3]
# Change here to only test a particular MLP type
linear_full_name = {'kp': 'Kolen-Pollack', 'fa': 'Feedback Alignment', 'bp': 'Backpropagation'}
linear_dict = {'fa': FeedbackAlignmentMLP, 'kp': KolenPollackMLP, 'bp': MLP}
bp_param_names = ['fc.0.weight', 'fc.0.bias', 'fc.2.weight', 'fc.2.bias', 'fc.4.weight', 'fc.4.bias']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, a=5**0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
def preprocess_state(state):
    agent_state = torch.tensor(state['agent'], dtype=torch.float32, requires_grad=True)
    target_state = torch.tensor(state['target'], dtype=torch.float32, requires_grad=True)
    return torch.concat([agent_state, target_state])

def select_action(state, epsilon, policy_net):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        return policy_net(state).argmax().item()

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

    state_batch = torch.stack(state_batch)
    action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32, requires_grad=True)
    next_state_batch = torch.stack(next_state_batch)
    done_batch = torch.tensor(done_batch, dtype=torch.float32, requires_grad=True)

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    # print("Backwarded")
    optimizer.step()
    return loss.item()

def train_dqn(linear_type, seed):

    set_seed(seed)
    env = GridWorldEnv(render_mode=None)
    model = linear_dict[linear_type]

    policy_net = model(state_dim, hidden_sizes, action_dim)
    target_net = model(state_dim, hidden_sizes, action_dim)
    policy_net.apply(init_weights)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.SGD(policy_net.parameters(), lr=0.01)
    memory = deque(maxlen=memory_size)

    epsilon = 1.0
    losses = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        total_loss = 0
        loss_num = 0  # optimize_model may return None

        for t in range(100):
            action = select_action(state, epsilon, policy_net)
            next_state, reward, terminated, _, _ = env.step(action)
            next_state = preprocess_state(next_state)
            memory.append((state, action, reward, next_state, terminated))

            state = next_state
            total_reward += reward

            loss = optimize_model(memory, policy_net, target_net, optimizer)
            if loss is not None:
                loss_num += 1
                total_loss += loss

            if terminated:
                break

        if loss_num > 0:  # loss_num = 0 if terminated within batch_size steps
            losses.append(total_loss / loss_num)
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}")

    torch.save(policy_net.state_dict(), os.path.join("output", f"dqn_gridworld_{linear_type}.pth"))
    env.close()
    return losses



def load_run_and_render(linear_type):
    env = GridWorldEnv(render_mode="human")

    model = linear_dict[linear_type](state_dim, hidden_sizes, action_dim)
    model.load_state_dict(torch.load(os.path.join("output", f"dqn_gridworld_{linear_type}.pth")))

    state, _ = env.reset()
    state = preprocess_state(state)
    for _ in range(100):
        action = select_action(state, epsilon_min, model)  # Use minimum epsilon for testing
        print(action)
        next_state, reward, terminated, _, _ = env.step(action)
        state = preprocess_state(next_state)
        env.render()
        if terminated:
            state, _ = env.reset()
            state = preprocess_state(state)

    env.close()


def plot_models_training_loss(model_losses, ax):
    for linear_type in linear_types:
        ax.plot(model_losses[linear_type], label=linear_full_name[linear_type])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Training Loss Curve')
    ax.legend()

def compute_cosine_similarity(data1, data2):
    data1 = data1.reshape(-1)
    data2 = data2.reshape(-1)
    numerator = np.dot(data1, data2)
    denominator = (
            np.sqrt(np.dot(data1, data1)) * np.sqrt(np.dot(data2, data2))
    )
    cosine_sim = numerator / denominator
    return cosine_sim

def plot_models_cosine_similarity(params, ax):
    cosine_means = {'kp': [], 'fa': []}
    cosine_stdevs = {'kp': [], 'fa': []}

    for episode in range(1, num_episodes):
        bp_params = params['bp']
        for model_type in ['kp', 'fa']:
            similarities = []
            for param_name in bp_param_names:
                update_bp = params['bp'][episode][param_name] - params['bp'][episode - 1][param_name]
                update_model = params[model_type][episode][param_name] - params[model_type][episode - 1][param_name]
                similarity = compute_cosine_similarity(update_model.cpu().numpy(), update_bp.cpu().numpy())
                similarities.append(similarity)
            mean_similarity = np.mean(similarities)
            stdev_similarity = np.std(similarities)
            cosine_means[model_type].append(mean_similarity)
            cosine_stdevs[model_type].append(stdev_similarity)

    for model_type in ['kp', 'fa']:
        ax.plot(cosine_means[model_type], label=f"{linear_full_name[model_type]} Mean")
        ax.fill_between(range(num_episodes-1),
                        np.array(cosine_means[model_type]) - np.array(cosine_stdevs[model_type]),
                        np.array(cosine_means[model_type]) + np.array(cosine_stdevs[model_type]),
                        alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Cosine Similarity of Parameter Updates')
    ax.legend()



# In the main code
if __name__ == "__main__":
    # load_run_and_render('fa')

    model_losses = {}
    # model_params = {}
    for linear_type in linear_types:
        losses = train_dqn(linear_type, 42)
        model_losses[linear_type] = losses
        # model_params[linear_type] = params

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    plot_models_training_loss(model_losses, ax1)
    # plot_models_cosine_similarity(model_params, ax2)
    plt.tight_layout()
    plt.show()

    # plot_models_SNR(model_params)

