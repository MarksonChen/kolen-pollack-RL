import gym
from gym import spaces
import pygame
import numpy as np
import random


class WindyGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        self.last_action = None 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.wind_direction = self._action_to_direction[random.randint(0, 4)]
        self.wind_magnitude = 1

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        self.last_action = action
        wind_effect = self.wind_direction * self.wind_magnitude
        new_location = self._agent_location + direction + wind_effect
        self._agent_location = np.clip(new_location, 0, self.size - 1)
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        arrow_start = np.array([self.window_size - 50, self.window_size - 50])
        arrow_end = arrow_start + self.wind_direction * 30
        pygame.draw.line(
            canvas,
            (0, 255, 0),
            arrow_start,
            arrow_end,
            width=3,
        )
        arrow_head_size = 5
        perp_direction = np.array([-self.wind_direction[1], self.wind_direction[0]])
        pygame.draw.polygon(
            canvas,
            (0, 255, 0),
            [
                arrow_end,
                arrow_end - self.wind_direction * arrow_head_size - perp_direction * arrow_head_size,
                arrow_end - self.wind_direction * arrow_head_size + perp_direction * arrow_head_size
            ]
        )

        if self.last_action is not None:
            action_arrow_start = np.array([50, 50])
            action_arrow_end = action_arrow_start + self._action_to_direction[self.last_action] * 30
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                action_arrow_start,
                action_arrow_end,
                width=3,
            )
            action_arrow_head_size = 5
            action_perp_direction = np.array([-self._action_to_direction[self.last_action][1], self._action_to_direction[self.last_action][0]])  # Perpendicular direction for arrow head
            pygame.draw.polygon(
                canvas,
                (0, 0, 0),
                [
                    action_arrow_end,
                    action_arrow_end - self._action_to_direction[self.last_action] * action_arrow_head_size - action_perp_direction * action_arrow_head_size,
                    action_arrow_end - self._action_to_direction[self.last_action] * action_arrow_head_size + action_perp_direction * action_arrow_head_size
                ]
            )
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    env = WindyGridWorldEnv(render_mode="human")
    observation, info = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            observation, info = env.reset()
    env.close()
