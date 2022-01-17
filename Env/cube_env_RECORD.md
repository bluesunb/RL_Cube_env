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
DQNPolicy의 feature_extractor가 2차원 이상의 MultiDiscrete space를 지원하지 않는 문제
###Issue description
feature_extractor로서, 1. FlattenExtractor(default), 2. NatureCNN
- FlattenExtractor: 1차원이 아닌 nvec을 가지는 MultiDiscrete 지원하지 않음
  - BaseFeatureExtractor을 init하기 위한 feature_dim 계산할 때 단순히 sum(nvec)을 해서 
    feature_dim>0 에서 element ambiguous error를 발생


- CnnPolicy: Box space만 지원

###Solution
1. `CubeEnv.state`의 6개의 surface를 vertical하게 concat하여 sum(observation_space.nvec) 이 integer가 될 수 있도록 조정
2. `BaseFeatureExtractor`를 implement 하는 새로운 feature_extractor 작성

- feature_extractor
    DQNPolicy의 q_net을 만들 때, mlp로 만들며, 
    feature_extractor.feature_dim: input_dim, action_dim: output_dim으로 하는 q_net을 생성한다. - - -

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
`evaluate_policy`에서 observation shape가 맞지 않는다는 오류가 발생하는 문제
###Issue description
[IR #1](#ir-1)의 solution에 따라 새로운 feature_extractor를 정의했으며 
결과적으로 feature는 Tensor[shape=(6,9)]이다.<br>
그러나 stable_baselines3에서는 env의 vectorization (여러개의 env를 한번에 학습하기 위해 병렬화 하는 것)을 위해 
`feature_extractor`를 통과한 observation이 1차원이길 원하나보다. (`BaseModel.obs_to_tensor()`)
###Solution
1. feature_extractor 구조를 다시 수정하여 최종 feature가 1차원이 되도록 만듬.
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
evaluate_policy를 통해 검증 시 observation이 features_extractor에 의해 변환되지 않는 문제
###Issue description
feature_extractor를 통해 observation을 적절히 변환하였지만, evaluate_policy에서는 적용되지 않았다. <br>
이는 <br>
`evaluate_policy() -> DQN.predict() -> self.policy.predict()` 로 이어지면서, <br>
`predict()` 메소드는 `DQNPolicy`에는 정의돼 있지 않고 (`_predict()`만 정의돼 있음) `BasePolicy`에만 정의돼 있어
`BasePolicy.predict()`로 bound 되기 때문이다.<br>여기서 `BasePolicy.predict()`
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
여기서 `obs_to_tensor()` 메소드에서 shape 문제가 일어난다.
###Solution
1. observation을 아예 1차원이 되도록 바꾼다. <br>
   이는, env의 vectorization을 지원하기 위해서인지 몰라도 2차원의 observation이 들어오면 
   이를 각각의 row가 각각의 env에서의 obs로 고려하는 듯 하다. <br>
    -> feature_extractor로 1차원으로 표현된 6x3x3 state를 적절히 처리해야 함

### Note
#### 한 개의 observation이 1차원이여야 하는 이유
`BasePolicy.predict() -> BaseModel.obs_to_tensor() -> utils.is_vectorized_observation()` <br>
이 때 `is_vectorized_observation`은 env vectorization 지원문제로 `discrete`, '`MultiDiscrete' 는 각각
1개의 env에 대해 0차원, 1차원 obs를 가질 것을 요구한다. <br>