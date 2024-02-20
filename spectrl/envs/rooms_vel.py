import numpy as np
import gym
import math
import copy


class VelRoomsEnv(gym.Env):

    # grid_params: GridParams
    def __init__(
        self, grid_params, init_dist, predicate, constraints, max_timesteps=1000
    ):
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
            self.init_dist = gym.spaces.Box(low=np.array([3, 3]), high=np.array([5, 5]), dtype=np.float32)
        self.predicate = copy.deepcopy(predicate)
        self.constraints = copy.deepcopy(constraints)
        self.max_timesteps = max_timesteps

        self.observation_dim = 2
        self.action_dim = 2

        # set the initial state
        self.reset()

    def reset(self):
        self.steps = 0
        self.state = self.init_dist.sample()
        return self.state

    def step(self, action):
        action = np.clip(action, -1, 1)
        next_state = self.state + 0.1 * action
        reach_reward = self.predicate(next_state)
        safe_reward = min([constraint(next_state) for constraint in self.constraints])
        violation_reward = 0 if self.path_clear(self.state, next_state) else 0

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

    # Check if straight line joining s1 and s2 does not pass through walls
    # s1 is assumed to be a legal state
    # we are assuming that wall size exceeds maximum action size
    # also assuming that door regions are small compared to rooms
    def path_clear(self, s1, s2):

        params = self.grid_params

        # find rooms of the states
        r1 = (s1 // params.partition_size).astype(np.int)
        r2 = (s2 // params.partition_size).astype(np.int)

        # find relative positions within rooms
        p1 = s1 - (r1 * params.partition_size)
        p2 = s2 - (r2 * params.partition_size)

        if not self.is_state_legal(s2, r2, p2):
            return False

        # both states are inside the same room (not in the door area)
        if (
            p1[0] <= params.room_size[0]
            and p1[1] <= params.room_size[1]
            and p2[0] <= params.room_size[0]
            and p2[1] <= params.room_size[1]
        ):
            return True
        # both states in door area
        if (p1[0] > params.room_size[0] or p1[1] > params.room_size[1]) and (
            p2[0] > params.room_size[0] or p2[1] > params.room_size[1]
        ):
            return True

        # swap to make sure s1 is in the room and s2 is in the door area
        if p2[0] <= params.room_size[0] and p2[1] <= params.room_size[1]:
            p1, p2 = p2, p1
            r1, r2 = r2, r1
            s1, s2 = s2, s1

        # four cases to consider
        if p2[0] > params.room_size[0]:
            # s1 is above s2
            if (r1 == r2).all():
                return self.check_vertical_intersect(p1, p2, params.room_size[0])
            # s1 is below s2
            else:
                return self.check_vertical_intersect(
                    (s1[0], p1[1]),
                    (s2[0], p2[1]),
                    (r2[0] + 1) * params.partition_size[0],
                )
        else:
            # s1 is left of s2
            if (r1 == r2).all():
                return self.check_horizontal_intersect(p1, p2, params.room_size[1])
            # s1 is right of s2
            else:
                return self.check_horizontal_intersect(
                    (p1[0], s1[1]),
                    (p2[0], s2[1]),
                    (r2[1] + 1) * params.partition_size[1],
                )

    # check if the state s is a legal state that is within the grid and not inside any wall area
    # r is the room of the state
    # p is the relative position within the room
    def is_state_legal(self, s, r, p):
        params = self.grid_params

        # make sure state is within the grid
        if not params.grid_region.contains(s):
            return False
        if r[0] >= params.size[0] or r[1] >= params.size[1]:
            return False

        # make sure state is not inside any wall area
        if p[0] <= params.room_size[0] and p[1] <= params.room_size[1]:
            return True
        elif (
            p[0] > params.room_size[0]
            and p[1] >= params.hdoor[0]
            and p[1] <= params.hdoor[1]
        ):
            return params.graph[params.get_index(r)][2]
        elif (
            p[1] > params.room_size[1]
            and p[0] >= params.vdoor[0]
            and p[0] <= params.vdoor[1]
        ):
            return params.graph[params.get_index(r)][3]
        else:
            return False

    # check if line from s1 to s2 intersects the horizontal axis at a point inside door region
    # horizontal coordinates should be relative positions within rooms
    def check_vertical_intersect(self, s1, s2, x):
        y = ((s2[1] - s1[1]) * (x - s1[0]) / (s2[0] - s1[0])) + s1[1]
        return self.grid_params.hdoor[0] <= y and y <= self.grid_params.hdoor[1]

    # check if line from s1 to s2 intersects the vertical axis at a point inside door region
    # vertical coordinates should be relative positions within rooms
    def check_horizontal_intersect(self, s1, s2, y):
        x = ((s2[0] - s1[0]) * (y - s1[1]) / (s2[1] - s1[1])) + s1[0]
        return self.grid_params.vdoor[0] <= x and x <= self.grid_params.vdoor[1]
