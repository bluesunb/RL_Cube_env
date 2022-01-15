#dqn_train Issue Record
- - -
## IR #1
Sat Jan  8 19:14:38 2022
<details>
<summary> <strong>Code</strong></summary>

```python
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from Sample.Cube.cube_env import CubeEnv
from Sample.Cube.cube_feature_extractor import CubeFeatureExtractor


env = CubeEnv(3)
check_env(env)
policy_kwargs = {'features_extractor_class': CubeFeatureExtractor,
                 'features_extractor_kwargs':
                     {'features_dim': 32*6}
                 }
model = DQN(MlpPolicy, env, verbose=1, policy_kwargs={'features_extractor_class': CubeFeatureExtractor})

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

model.learn(10)
```

</details>

###Issue name
Training 성능의 저조
###Issue description
1. QNetwork의 구조 문제(`net_arch`)인 경우
2. FeaturesExtractor의 구조 문제인 경우
3. observation_space의 문제인 경우
###Solution
1. QNetwork의 구조를 바꿔본다.
2. FeaturesExtractor의 구조를 바꾼다.
3. observation_space를 `MultiDiscrete`에서 `Box`로 바꾼다.
    - Note: 
      `Multidiscrete` obs가 `utils.preprocess_obs()`로부터 one-hot encoding이 되는 것으로 보아, 
       Multi-observation 에 적합한 space일 것이라고 추측.  
       따라서 multi-value를 자연스럽게 지원하는 Box를 int type으로 사용해볼 수 있을 것.
    
### Note
[stable_baselines3.md](Sample/Cube/baselines3.md)를 통해 learning process, parameter 참고.- - -
## IR #2
Fri Jan 14 10:27:29 2022
<details>
<summary> <strong>Code</strong> </summary>

```python
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env

from Sample.Cube.cube_env import CubeEnv
from Sample.Cube.cube_feature_extractor import CubeFeatureExtractor


env = CubeEnv(3)
check_env(env)

dqn_kwargs = {'policy': MlpPolicy,
              'env': env,
              'learning_rate': 1e-4,
              'buffer_size': 1e6,
              'learning_starts': 5e4,
              'batch_size': 32,
              'tau': 1.0,
              'gamma': 0.99,
              'train_freq': 4,
              'gradient_steps': 1,
              'replay_buffer_class': None,
              'replay_buffer_kwargs': None,
              'optimize_memory_usage': False,
              'target_update_interval': 1e4,
              'exploration_fraction': 0.1,
              'exploration_initial_eps': 1.0,
              'exploration_final_eps': 0.01,
              'max_grad_norm': 10.0,
              'tensorboard_log': None,
              'create_eval_env': False,
              'policy_kwargs': None,
              'verbose': 0,
              'seed': None,
              'device': 'auto',
              '_init_setup_model': True}

policy_kwargs = {
    'features_extractor_class': CubeFeatureExtractor,
    'features_extractor_kwargs': {'features_dim': 32*6},
    'net_arch': [256, 256, 192, 64],
}

import torch.optim as optim

dqn_kwargs['tau'] = 0.99
dqn_kwargs['gradient_steps'] = -1
dqn_kwargs['verbose'] = 1
dqn_kwargs['learning_rate'] = optim.lr_scheduler.LambdaLR() # ????

model = DQN(MlpPolicy, env, verbose=1, policy_kwargs={'features_extractor_class': policy_kwargs})
model.learn(10)

```

</details>

###Issue name
stable_baselines3 training에서 lr_scheduler를 설정하는 문제
###Issue description
- DQN Model에 lr을 따로 넘기고, DQNPolicy에 optimizer_class를 따로 넘긴다.
- lr은 인스턴스로 넘겨야 하고, optimizer는 class type으로 넘겨야 해서 lr_scheduler 인스턴스 초기화에 필요한 optimizer를 지정할 수 없다.  
    `optim.lr_scheduler.StepLR(optimizer=optimizer, ...)`
###Solution
- model kwargs parameter로 `learning_rate`를 적절한 function으로 넘겨주면 된다.  
  -> `learning_rate`가 `_current_progress_remaining` (전체 traning 중에서 현재) 

### Note
**Model Learning Process**
  - `DQN.learn() -> ... DQN.train() -> BaseAlgorithm._update_learning_rate() -> utils.update_learning_rate()`
  
  
<details>
<summary>
  <code>DQN.train()</code>
</summary>

```python
def train(self, gradient_steps: int, batch_size: int = 100) -> None:
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer) # <- !!!
    ...
```
</details>

<details>
<summary>
  <code>BaseAlgorithm._update_learning_rate()</code>
</summary>

```python
from stable_baselines3.common.base_class import *

def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
    """
    Update the optimizers learning rate using the current learning rate schedule
    and the current progress remaining (from 1 to 0).
    :param optimizers:
        An optimizer or a list of optimizers.
    """
    # Log 기록
    self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))
    
    if not isinstance(optimizers, list):    # optimizers가 list가 아니면 list로 만듦
        optimizers = [optimizers]
    for optimizer in optimizers:    # optimizers:list 에 있는 모든 optimizer에 대해 lr update
        update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))
```
</details>

<details>
<summary>
  <code>utils.update_learning_rate()</code>
</summary>

```python
import torch as th

def update_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:

    # optimizer의 모든 parameters 에 대해 lr을 learning_rate로 세팅
    for param_group in optimizer.param_groups:  
        param_group["lr"] = learning_rate
```
</details>