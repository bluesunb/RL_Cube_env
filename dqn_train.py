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