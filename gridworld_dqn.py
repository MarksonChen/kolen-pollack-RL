import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np

from gridworld import GridWorldEnv
from MLPs.kolen_pollack import KolenPollackMLP
from MLPs.feedback_alignment import FeedbackAlignmentMLP
from MLPs.mlp import MLP
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
num_episodes = 500
episode_steps = 100
hidden_sizes = [128]

env = GridWorldEnv(render_mode=None)
state_dim = np.prod(env.observation_space['agent'].shape) * 2  # Agent and Target
action_dim = env.action_space.n


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
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, a=5**0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

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

def gradient_dictionary_to_numpy(models_gradients):
    """
    input: model_gradients: {linear_type: gradients_dict}
                gradients_dict: {param_names: [num_episodes x (<episode_steps)
                        x (gradients tensor)]}
    output: {linear_type: 4D numpy array}
                4D numpy array: [num_params x num_episodes x (<episode_steps)
                        x (gradients tensor)]
    """
    models_grad_np = {}
    for linear_type in linear_types:
        param_grads = models_gradients[linear_type]
        for key in param_grads:
            if "feedback" in key:
                del param_grads[key]

        num_params = len(param_grads)
        max_flattened_size = max(param.numel() for episodes in param_grads.values() for steps in episodes for param in steps)
        grads_array = np.full((num_params, num_episodes, episode_steps, max_flattened_size), np.nan)

        # Gradient dictionary to 4D numpy array
        for i, (key, episodes) in enumerate(param_grads.items()):
            for j, steps in enumerate(episodes):
                for k, param in enumerate(steps):
                    flattened_tensor = param.flatten().numpy()
                    grads_array[i, j, k, :len(flattened_tensor)] = flattened_tensor

        models_grad_np[linear_type] = grads_array
    return models_grad_np


def map_state_dict_keys(state_dict, model_type):
    if model_type == 'bp':  # For MLP
        key_mapping = {
            "weights.0": "model.0.weight",
            "biases.0": "model.0.bias",
            "weights.1": "model.2.weight",
            "biases.1": "model.2.bias"
        }
    elif model_type == 'kp':  # For KolenPollackMLP
        key_mapping = {
            "model.0.weight": "weights.0",
            "model.0.bias": "biases.0",
            "model.2.weight": "weights.1",
            "model.2.bias": "biases.1"
        }
    elif model_type == 'fa':  # For FeedbackAlignmentMLP
        key_mapping = {
            "weights.0": "model.0.weight",
            "biases.0": "model.0.bias",
            "weights.1": "model.2.weight",
            "biases.1": "model.2.bias"
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    mapped_state_dict = {}
    for old_key, new_key in key_mapping.items():
        # print(model_type, old_key in state_dict, old_key, state_dict.keys())
        if old_key in state_dict:
            mapped_state_dict[new_key] = state_dict[old_key]

    return mapped_state_dict

def train_dqn(linear_type, seed, initial_params=None, get_gradients=False):

    set_seed(seed)
    env = GridWorldEnv(render_mode=None)
    model = linear_dict[linear_type]

    policy_net = model(state_dim, hidden_sizes, action_dim)
    target_net = model(state_dim, hidden_sizes, action_dim)
    init_params = {k: v.clone().detach() for k, v in policy_net.state_dict().items()}
    if initial_params is not None:
        policy_net.load_state_dict(initial_params)
        target_net.load_state_dict(initial_params)
    else:
        policy_net.apply(init_weights)
        target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.SGD(policy_net.parameters(), lr=0.01)
    memory = deque(maxlen=memory_size)

    epsilon = 1.0
    losses = []


    if get_gradients:
        gradients = {}
        for name, param in policy_net.named_parameters():
            if param.requires_grad:
                gradients[name] = []
    else:
        gradients = None

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        total_loss = 0
        loss_num = 0  # optimize_model may return None

        if get_gradients:
            for name, param in policy_net.named_parameters():
                gradients[name].append([])

        for t in range(episode_steps):
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
                if get_gradients:
                    # gradients[-1].append([param.grad for name, param in policy_net.named_parameters()])
                    for name, param in policy_net.named_parameters():
                        gradients[name][-1].append(param.grad)

            if terminated:
                break

        if loss_num > 0:  # loss_num = 0 if terminated within batch_size steps
            losses.append(total_loss / loss_num)
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}")

    # torch.save(policy_net.state_dict(), os.path.join("output", f"dqn_gridworld_{linear_type}.pth"))
    env.close()
    return losses, gradients, init_params

def remove_feedback_gradients(param_grads):
    keys_to_delete = [key for key in param_grads.keys() if "feedback" in key]
    for key in keys_to_delete:
        del param_grads[key]
    return param_grads

def train_models(model_types=None):
    if model_types is None:
        model_types = linear_types
    models_losses = {}
    models_gradients = {}

    initial_params = None
    for linear_type in model_types:
        initial_params = map_state_dict_keys(initial_params, linear_type) if initial_params is not None else None
        losses, param_grads, init_params = train_dqn(linear_type, 42,
                    initial_params=initial_params, get_gradients=True)
        initial_params = {k: v.clone().detach() for k, v in init_params.items()}

        models_losses[linear_type] = losses
        models_gradients[linear_type] = remove_feedback_gradients(param_grads)
    return models_losses, models_gradients

# In the main code
def main():
    # model_losses = {}
    # for linear_type in linear_types:
    #     losses, _ = train_dqn(linear_type, 42)
    #     model_losses[linear_type] = losses

    models_losses, models_gradients = train_models(model_types=['kp', 'bp'])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    plot_models_training_loss(models_losses, ax1)
    # plot_models_cosine_similarity(models_gradients, ax2)
    # plot_models_snr(models_gradients, ax3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

