import gym
import gym.spaces as spaces
import numpy as np

from typing import Optional

from stable_baselines3.common.type_aliases import GymObs

from Sample.Cube.Env.cube import Cube


class CubeEnv(gym.Env):
    def __init__(self, size=2, max_steps=50, eval_env=False):
        super(CubeEnv, self).__init__()
        self.size = size
        self.cube = Cube(size)
        self.state = self.cube.state
        self.max_steps = max_steps
        self.step_counter = 0
        self.before_reward = 0

        # self.observation_space = spaces.MultiDiscrete(np.zeros_like(self.state) + 6)
        self.observation_space = spaces.MultiDiscrete((np.zeros_like(self.state) + 6).flatten())
        # self.action_space = spaces.Discrete(12)
        # action: ([up, down, left, right, front, back], [clockwise_false, clockwise_true])
        # => u,d,l,r,f,b,u', ...
        self.action_space = spaces.Discrete(12)

        # hyper-parameters for learning process
        self.step_penalty = 1       # penalty for step progression
        self.fractional_reward = False    # whether to make reward between -1.0 ~ 1.0

        # for evaluation
        self.eval_env = eval_env

    @property
    def state(self):
        return self.cube.state

    @state.setter
    def state(self, state):
        assert self.cube.state.shape == state.shape, 'state size is different!'
        self.cube.state = state

    def get_obs(self) -> GymObs:
        # return self.state.copy()
        return self.state.flatten()

    def reset(self) -> GymObs:
        self.cube.reset()
        self.cube.shuffle(length=5)
        self.step_counter = 0

        obs = self.get_obs()
        self.before_reward = self.get_reward(obs)
        return obs

    def step(self, action: int, render: Optional[bool] = None):
        done, info = False, {'msg': 'On Going'}

        clockwise, surface = divmod(action, 6)
        self.cube.move(surface, clockwise)
        self.step_counter += 1

        obs = self.get_obs()
        now_reward = self.get_reward(obs)
        reward = now_reward - self.before_reward
        self.before_reward = now_reward

        if now_reward == (self.cube.size**2)*6:
            done = True
            info['msg'] = 'Complete'
        elif self.step_counter >= self.max_steps:
            done = True
            info['msg'] = 'Max Step'

        # apply step penalty and fractional_reward
        reward -= self.step_penalty

        if self.fractional_reward:
            reward /= (4*self.size)

        render = True if render or (self.eval_env and render is None) else False
        if render:
            self.render(mode='step',
                        surface=surface,
                        clockwise=clockwise,
                        reward=reward,
                        done=done,
                        info=info)

        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        if mode=='step':
            print(f'[Step:{self.step_counter}] '
                  f'surface: {kwargs["surface"]} clockwise: {kwargs["clockwise"]}')
            self.cube.render()
            print(f'reward: {kwargs["reward"]}, now_r: {self.before_reward}, done: {kwargs["done"]}, info: {kwargs["info"]}')

        elif mode=='human':
            self.cube.render()
            print()

    @staticmethod
    def obs_to_state_form(obs) -> np.ndarray:
        # if not self.observation_space.shape == self.state.shape:
        #     return obs.reshape(self.state.shape)
        # else:
        #     return obs
        if len(obs.shape) != 3 or obs.shape[0] != 6:
            size = int(np.sqrt(obs.size/6))
            return obs.reshape(6, size, size)
        else:
            return obs

    @staticmethod
    def get_reward(obs) -> float:
        state = CubeEnv.obs_to_state_form(obs)
        return float(sum([np.sum(state[i] == i) for i in range(6)]))


class EvalCubeEnv(CubeEnv):
    def __init__(self, size, max_steps=50, eval_env=False):
        super().__init__(size, max_steps, eval_env)

    def reset(self):
        self.cube.reset()

        move_seq = [(0,1), (1,0), (2,1), (3,0), (4,1)]
        for surface_idx, direction in move_seq:
            self.cube.move(surface_idx, bool(direction))

        self.step_counter = 0

        obs = self.get_obs()
        self.before_reward = self.get_reward(obs)
        return obs
