import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from itertools import product


class Cube:
    def __init__(self, size=2):
        """
        front: 0, top: 1, left: 2, bottom: 3, right: 4, back: 5
        To satisfy symmetry, orientation of surfaces are different: [up, down left, right, down, right]
        """
        self.size = size
        self.state = np.array([np.zeros((self.size, self.size)) + i for i in range(6)], dtype=int)
        self._connectivity = [(1, 2, 3, 4), (0, 4, 5, 2), (5, 3, 0, 1), (4, 5, 2, 0), (3, 5, 1, 0), (2, 1, 4, 3)]

    def get_obs(self):
        return self.state.copy()

    def move(self, surface: int, clockwise: bool, render=False):
        """
        move rotate cube surface
        :param surface: surface idx
        :param clockwise:
        :param render:
        :return:
        """
        self.state[surface] = \
            self.state[surface].T[:, ::-1] if clockwise else self.state[surface].T[::-1]

        conn = self._connectivity[surface]
        edges = [
            self.state[conn[0], 0, :].copy(),
            self.state[conn[1], -1, ::-1].copy(),
            self.state[conn[2], :, 0].copy(),
            self.state[conn[3], ::-1, -1].copy()
        ]
        if not clockwise:
            edges = [np.flip(arr) for arr in edges]

        self.state[conn[0], 0, :] = edges[1] if clockwise else edges[-1]
        self.state[conn[1], -1, :] = edges[2] if clockwise else edges[0]
        self.state[conn[2], :, 0] = edges[3] if clockwise else edges[1]
        self.state[conn[3], :, -1] = edges[0] if clockwise else edges[2]

        if render:
            print(f'surface: {surface}\t clockwise: {clockwise}')
            self.render()

    def shuffle(self, length: int = 6):
        shuffle_surface = np.random.randint(0, 6, size=length)
        shuffle_clockwise = np.random.randint(0, 2, size=length)
        for surface, clockwise in zip(shuffle_surface, shuffle_clockwise):
            self.move(surface, bool(clockwise))

    def reset(self):
        self.state = np.array([np.zeros((self.size, self.size)) + i for i in range(6)])

    def render(self, mode='human') -> None:
        def colorize_print(s):
            if isinstance(s, np.ndarray):
                return f'|{" ".join(colorize_print(sin) for sin in s)}|'
            i = int(s)
            if i < 6:
                return '\033[' + str(31 + i) + 'm' + str(i) + '\033[0m'
            return str(i)

        # surface 1
        surf1 = self.state[1][::-1, ::-1]
        for line in surf1:
            print(' ' * (1 + self.size * 2) + colorize_print(line))

        # surface 2, 0, 4 || 5
        surf2 = self.state[2].T[::-1]
        surf0 = self.state[0]
        surf4 = self.state[4][::-1, ::-1]
        surf5 = self.state[5].T[:, ::-1]
        for l1, l2, l3, l4 in zip(surf2, surf0, surf4, surf5):
            print(colorize_print(l1) + colorize_print(l2) + colorize_print(l3), end=' |')
            print(colorize_print(l4))

        # surface 3
        surf3 = self.state[3].T[:, ::-1]
        for line in surf3:
            print(' ' * (1 + self.size * 2) + colorize_print(line))