import os

import imageio

from gridworld import GridWorldEnv
from gridworld_dqn import *

env = GridWorldEnv(render_mode="rgb_array")
frames = []
state, _ = env.reset()
state = preprocess_state(state)
linear_type = 'kp'
policy_net = linear_dict[linear_type](state_dim, hidden_sizes, action_dim)
policy_net.load_state_dict(torch.load(os.path.join("output", f"dqn_gridworld_{linear_type}.pth")))
for _ in range(100):
    action = select_action(state, 0, policy_net)
    next_state, reward, terminated, _, _ = env.step(action)
    state = preprocess_state(next_state)
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    if terminated:
        state, _ = env.reset()
        state = preprocess_state(state)
env.close()
imageio.mimsave(os.path.join("output", "dqn_gridworld.gif"), frames, fps=10)
