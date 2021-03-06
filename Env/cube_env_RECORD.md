#cube_env Issue Record
- - -

## IR #1
Thu Jan  6 20:49:51 2022
<details>
<summary> <strong>Code</strong> </summary>

```python
import gym
import gym.spaces as spaces
import numpy as np

from stable_baselines3.common.type_aliases import GymEnv, GymObs, GymStepReturn

from Sample.Cube.Env.cube import Cube


class CubeEnv(gym.Env):
    def __init__(self, size=2, max_steps=50):
        super(CubeEnv, self).__init__()
        self.cube = Cube(size)
        self.state = self.cube.state
        self.max_steps = max_steps
        self.step_counter = 0
        self.before_reward = 0

        # self.observation_space = spaces.MultiDiscrete(np.zeros_like(self.state) + 6)
        self.observation_space = spaces.Tuple(
            [spaces.MultiDiscrete(np.zeros((size, size)) + 6)] * 6)
        # self.action_space = spaces.Discrete(12)
        # action: ([up, down, left, right, front, back], [clockwise_false, clockwise_true])
        # => u,d,l,r,f,b,u', ...
        self.action_space = spaces.Discrete(12)

    @property
    def state(self):
        return self.cube.state

    @state.setter
    def state(self, state):
        assert self.cube.state.shape == state.shape, 'state size is different!'
        self.cube.state = state

    def get_obs(self) -> GymObs:
        return tuple(surface.copy() for surface in self.cube.state)

    def reset(self) -> GymObs:
        self.cube.reset()
        self.cube.shuffle(length=6)
        self.step_counter = 0

        obs = self.get_obs()
        self.before_reward = self.get_reward(obs)
        return obs

    def step(self, action: int, render: bool = False):
        done, info = False, {'msg': 'On Going'}

        clockwise, surface = divmod(action, 6)
        self.cube.move(surface, clockwise)
        self.step_counter += 1

        obs = self.get_obs()
        now_reward = self.get_reward(obs)
        if now_reward == (self.cube.size ** 2) * 6:
            done = True
            info['msg'] = 'Complete'
        elif self.step_counter >= self.max_steps:
            done = True
            info['msg'] = 'Max Step'

        if render:
            self.render(mode='step',
                        surface=action[0],
                        clockwise=bool(action[1]),
                        reward=now_reward - self.before_reward,
                        done=done,
                        info=info)
        return obs, now_reward - self.before_reward, done, info

    def render(self, mode='human', **kwargs):
        if mode == 'step':
            print(f'[Step:{self.step_counter}] '
                  f'surface: {kwargs["surface"]} clockwise: {kwargs["clockwise"]}')
            self.cube.render()
            print(f'reward: {kwargs["reward"]} done: {kwargs["done"]}, info: {kwargs["info"]}')

        elif mode == 'human':
            self.cube.render()
            print()

    @staticmethod
    def get_reward(state) -> float:
        return float(sum([np.sum(state[i] == i) for i in range(6)]))


from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy, MlpPolicy
from stable_baselines3.common.env_checker import check_env

env = CubeEnv(3)
check_env(env)
model = DQN(MlpPolicy, env, verbose=1)
```
```
Traceback (most recent call last):
  File "C:/Users/bluesun/PycharmProjects/RL_project/Sample/Cube/cube_env.py", line 95, in <module>
    model = DQN(MlpPolicy, env, verbose=1)
  File "C:\Users\bluesun\anaconda3\envs\rl\lib\site-packages\stable_baselines3\common\torch_layers.py", line 23, in __init__
    assert features_dim > 0
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

</details>

###Issue name
DQNPolicy??? feature_extractor??? 2?????? ????????? MultiDiscrete space??? ???????????? ?????? ??????
###Issue description
feature_extractor??????, 1. FlattenExtractor(default), 2. NatureCNN
- FlattenExtractor: 1????????? ?????? nvec??? ????????? MultiDiscrete ???????????? ??????
  - BaseFeatureExtractor??? init?????? ?????? feature_dim ????????? ??? ????????? sum(nvec)??? ?????? 
    feature_dim>0 ?????? element ambiguous error??? ??????


- CnnPolicy: Box space??? ??????

###Solution
1. `CubeEnv.state`??? 6?????? surface??? vertical?????? concat?????? sum(observation_space.nvec) ??? integer??? ??? ??? ????????? ??????
2. `BaseFeatureExtractor`??? implement ?????? ????????? feature_extractor ??????

- feature_extractor
    DQNPolicy??? q_net??? ?????? ???, mlp??? ?????????, 
    feature_extractor.feature_dim: input_dim, action_dim: output_dim?????? ?????? q_net??? ????????????. - - -

---
## IR #2
Thu Jan  6 23:53:47 2022
<details>
<summary> <strong>Code</strong> </summary>

```python
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy, MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy

import gym
import torch as th

from Sample.Cube.Env.cube_env import CubeEnv


class CubeFeatureExtractor(BaseFeaturesExtractor):
    """
    for cube, observation_space.shape: (6, 3, 3)
    process:
        (6,3,3) -> (6,9) -> (6, features_dim)
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 32):
        super().__init__(observation_space, features_dim)
        initial_dim = observation_space.sample().shape[-1]
        self.layers = th.nn.Sequential(
            th.nn.Flatten(),
            th.nn.Linear(initial_dim ** 2, features_dim),
            th.nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.layers(observations)


env = CubeEnv(3)
check_env(env)
model = DQN(MlpPolicy, env, verbose=1, policy_kwargs={'features_extractor_class': CubeFeatureExtractor})

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
```

```
ValueError                                Traceback (most recent call last)
<ipython-input-13-7f0e4fccb2ae> in <module>()
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
/usr/local/lib/python3.7/dist-packages/stable_baselines3/common/utils.py in is_vectorized_observation(observation, observation_space)
    return is_vec_obs_func(observation, observation_space)
/usr/local/lib/python3.7/dist-packages/stable_baselines3/common/utils.py in is_vectorized_multidiscrete_observation(observation, observation_space)
    + f"(n_env, {len(observation_space.nvec)}) for the observation shape."

ValueError: Error: Unexpected observation shape (1, 6, 3, 3) for MultiDiscrete environment, please use (6,) or (n_env, 6) for the observation shape.
```

</details>

###Issue name
`evaluate_policy`?????? observation shape??? ?????? ???????????? ????????? ???????????? ??????
###Issue description
[IR #1](#ir-1)??? solution??? ?????? ????????? feature_extractor??? ??????????????? 
??????????????? feature??? Tensor[shape=(6,9)]??????.<br>
????????? stable_baselines3????????? env??? vectorization (???????????? env??? ????????? ???????????? ?????? ????????? ?????? ???)??? ?????? 
`feature_extractor`??? ????????? observation??? 1???????????? ???????????????. (`BaseModel.obs_to_tensor()`)
###Solution
1. feature_extractor ????????? ?????? ???????????? ?????? feature??? 1????????? ????????? ??????.
- - -
## IR #3
Fri Jan  7 18:36:39 2022
<details>
<summary> <strong>Code</strong> </summary>

```python
import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy, MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy

import torch as th

from Sample.Cube.Env.cube_env import CubeEnv


class CubeFeatureExtractor(BaseFeaturesExtractor):
    """
    for cube, observation_space.shape: (6, 3, 3)
    process:
        (6,3,3) -> (6,9) -> (6, features_dim) -> (
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 32 * 6):
        if features_dim % 6:
            features_dim = int(round(features_dim / 6)) * 6
        super().__init__(observation_space, features_dim)
        initial_dim = observation_space.sample().shape[-1]
        self.layers = th.nn.Sequential(
            th.nn.Flatten(),
            th.nn.Linear(initial_dim ** 2, features_dim // 6),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.layers(observations)


env = CubeEnv(3)
check_env(env)
policy_kwargs = {'features_extractor_class': CubeFeatureExtractor,
                 'features_extractor_kwargs':
                     {'features_dim': 32 * 6}
                 }
model = DQN(MlpPolicy, env, verbose=1, policy_kwargs={'features_extractor_class': CubeFeatureExtractor})

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
```
```
Traceback (most recent call last):
  File "C:\Users\bluesun\anaconda3\envs\rl\lib\site-packages\IPython\core\interactiveshell.py", line 3437, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-2-b8ce443a0fba>", line 1, in <module>
    runfile('C:/Users/bluesun/PycharmProjects/RL_project/Sample/Cube/cube_env.py', wdir='C:/Users/bluesun/PycharmProjects/RL_project/Sample/Cube')
  File "C:\Program Files\JetBrains\PyCharm 2020.3.3\plugins\python\helpers\pydev\_pydev_bundle\pydev_umd.py", line 198, in runfile
    pydev_imports.execfile(filename, global_vars, local_vars)  # execute the script
  File "C:\Program Files\JetBrains\PyCharm 2020.3.3\plugins\python\helpers\pydev\_pydev_imps\_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "C:/Users/bluesun/PycharmProjects/RL_project/Sample/Cube/cube_env.py", line 129, in <module>
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
  File "C:\Users\bluesun\anaconda3\envs\rl\lib\site-packages\stable_baselines3\common\evaluation.py", line 85, in evaluate_policy
    actions, states = model.predict(observations, state=states, deterministic=deterministic)
  File "C:\Users\bluesun\anaconda3\envs\rl\lib\site-packages\stable_baselines3\dqn\dqn.py", line 225, in predict
    action, state = self.policy.predict(observation, state, mask, deterministic)
  File "C:\Users\bluesun\anaconda3\envs\rl\lib\site-packages\stable_baselines3\common\policies.py", line 323, in predict
    observation, vectorized_env = self.obs_to_tensor(observation)
  File "C:\Users\bluesun\anaconda3\envs\rl\lib\site-packages\stable_baselines3\common\policies.py", line 240, in obs_to_tensor
    vectorized_env = is_vectorized_observation(observation, self.observation_space)
  File "C:\Users\bluesun\anaconda3\envs\rl\lib\site-packages\stable_baselines3\common\utils.py", line 357, in is_vectorized_observation
    return is_vec_obs_func(observation, observation_space)
  File "C:\Users\bluesun\anaconda3\envs\rl\lib\site-packages\stable_baselines3\common\utils.py", line 280, in is_vectorized_multidiscrete_observation
    raise ValueError(
ValueError: Error: Unexpected observation shape (1, 6, 3, 3) for MultiDiscrete environment, please use (6,) or (n_env, 6) for the observation shape.
```
</details>

###Issue name
evaluate_policy??? ?????? ?????? ??? observation??? features_extractor??? ?????? ???????????? ?????? ??????
###Issue description
feature_extractor??? ?????? observation??? ????????? ??????????????????, evaluate_policy????????? ???????????? ?????????. <br>
?????? <br>
`evaluate_policy() -> DQN.predict() -> self.policy.predict()` ??? ???????????????, <br>
`predict()` ???????????? `DQNPolicy`?????? ????????? ?????? ?????? (`_predict()`??? ????????? ??????) `BasePolicy`?????? ????????? ??????
`BasePolicy.predict()`??? bound ?????? ????????????.<br>????????? `BasePolicy.predict()`
- `BasePolicy.predict()`
    ```python
    import numpy as np
    import torch as th
    from typing import Optional, Union, Tuple, Dict
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    
        self.set_training_mode(False)
        observation, vectorized_env = self.obs_to_tensor(observation) #ERROR ROSE
    
        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        
            ... #OMMITTED
    
        return actions, state
    ```
????????? `obs_to_tensor()` ??????????????? shape ????????? ????????????.
###Solution
1. observation??? ?????? 1????????? ????????? ?????????. <br>
   ??????, env??? vectorization??? ???????????? ??????????????? ????????? 2????????? observation??? ???????????? 
   ?????? ????????? row??? ????????? env????????? obs??? ???????????? ??? ??????. <br>
    -> feature_extractor??? 1???????????? ????????? 6x3x3 state??? ????????? ???????????? ???

### Note
#### ??? ?????? observation??? 1??????????????? ?????? ??????
`BasePolicy.predict() -> BaseModel.obs_to_tensor() -> utils.is_vectorized_observation()` <br>
??? ??? `is_vectorized_observation`??? env vectorization ??????????????? `discrete`, '`MultiDiscrete' ??? ??????
1?????? env??? ?????? 0??????, 1?????? obs??? ?????? ?????? ????????????. <br>