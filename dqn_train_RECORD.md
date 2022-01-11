#dqn_train Issue Record
- - -
## IR #1
Sat Jan  8 19:14:38 2022
<details>
<summary> Code</summary>

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
[stable_baselines3.md](Sample/Cube/baselines3.md)를 통해 learning process, parameter 참고.