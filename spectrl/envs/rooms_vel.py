import numpy as np
import gym
import math
import copy


class VelRoomsEnv(gym.Env):

    # grid_params: GridParams
    def __init__(
        self, grid_params, init_dist, predicate, constraints, max_timesteps=1000
    ):
        self.init_radius = 0.1
        self.grid_params = grid_params
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        if init_dist is not None:
            self.init_dist = gym.spaces.Box(
                low=init_dist[0], high=init_dist[1], dtype=np.float32
            )
        else:
            self.init_dist = gym.spaces.Box(
                low=np.array([0.5 - self.init_radius, 0.5 - self.init_radius]),
                high=np.array([0.5 + self.init_radius, 0.5 + self.init_radius]),
                dtype=np.float32,
            )
        self.predicate = copy.deepcopy(predicate)
        self.constraints = copy.deepcopy(constraints)
        self.max_timesteps = max_timesteps

        self._v_lines = [
            (3, [(0, 3)]),
            # (2, [(0, 1.2), (1.8, 2.2), (2.8, 3)]),
            # (1, [(0, 0.2), (0.8, 2.2), (1.8, 2.2), (2.8, 3)]),
            (2, [(0, 1.1), (1.9, 2.1), (2.9, 3)]),
            (1, [(0, 0.1), (0.9, 2.1), (1.9, 2.1), (2.9, 3)]),
            # (2, [(0, 0.2), (0.8, 1.2), (1.8, 2.2), (2.8, 3)]),
            # (1, [(0, 0.2), (0.8, 1.2), (1.8, 2.2), (2.8, 3)]),
            (0, [(0, 3)]),
        ]
        self._h_lines = [
            (3, [(0, 3)]),
            # (2, [(0, 0.2), (0.8, 1.2), (1.8, 2.2), (2.8, 3)]),
            # (1, [(0, 0.2), (0.8, 1.2), (1.8, 2.2), (2.8, 3)]),
            (2, [(0, 0.1), (0.9, 1.1), (1.9, 2.1), (2.9, 3)]),
            (1, [(0, 0.1), (0.9, 1.1), (1.9, 2.1), (2.9, 3)]),
            # (2, [(0, 0.2), (0.8, 1.2), (1.8, 2.2), (2.8, 3)]),
            # (1, [(0, 0.2), (0.8, 1.2), (1.8, 2.2), (2.8, 3)]),
            (0, [(0, 3)]),
        ]
        lw = 0.05
        self.unsafe_spaces = []
        for y, xlines in self._v_lines:
            for x1, x2 in xlines:
                self.unsafe_spaces.append(
                    gym.spaces.Box(
                        low=np.array([x1 - lw, y - lw]),
                        high=np.array([x2 + lw, y + lw]),
                        dtype=np.float32,
                    )
                )
        for x, ylines in self._h_lines:
            for y1, y2 in ylines:
                self.unsafe_spaces.append(
                    gym.spaces.Box(
                        low=np.array([x - lw, y1 - lw]),
                        high=np.array([x + lw, y2 + lw]),
                        dtype=np.float32,
                    )
                )

        # set the initial state
        self.reset()

    @property
    def observation_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.action_space.shape[0]

    @property
    def horizon(self):
        return self.max_timesteps

    def get_env_infos(self):
        return {}

    def set_seed(self, seed):
        return

    def reset(self):
        self.steps = 0
        self.state = self.init_dist.sample()
        return self.state

    def step(self, action):
        noise = np.random.uniform(low=-0.5, high=0.5)

        next_state = self.state + 0.1 * (action + noise)

        contain = False
        for interp in [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            for unsafe_space in self.unsafe_spaces:
                interp_state = self.state * interp + (1 - interp) * next_state
                contain = np.all(
                    np.logical_and(
                        interp_state >= unsafe_space.low,
                        interp_state <= unsafe_space.high,
                    )
                )
        reach_reward = self.predicate(next_state)
        safe_reward = min([constraint(next_state) for constraint in self.constraints])
        violation_reward = -100 if contain else 0

        self.state = next_state
        self.steps += 1
        done = self.steps > self.max_timesteps
        return self.state, reach_reward + safe_reward + violation_reward, done, {}

    def render(self):
        pass

    def get_sim_state(self):
        return self.state

    def set_sim_state(self, state):
        self.state = state
        return self.state

    def close(self):
        pass
