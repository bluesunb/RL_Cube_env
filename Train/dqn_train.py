from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env

from Sample.Cube.Env.cube_env import CubeEnv
from Sample.Cube.DQN.cube_feature_extractor import CubeFeatureExtractor


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

dqn_kwargs['policy'] = MlpPolicy
dqn_kwargs['env'] = env
dqn_kwargs['learning_rate'] = lambda r: 0.01 * (0.5**(10*(1-r)))    # r: current_progress_remaining
dqn_kwargs['buffer_size'] = int(1e4)     # size of replay_buffer
dqn_kwargs['learning_starts'] = 1000     # num of transitions to collect before training
dqn_kwargs['batch_size'] = 16       # size of mini-batch which comes from sampling
dqn_kwargs['train_freq'] = 1        # model update frequency = size of collecting rollout
dqn_kwargs['target_update_interval'] = 100      # target network update freq
dqn_kwargs['tau'] = 0.99    # soft update coefficient, tau: ratio reflecting non-target network
dqn_kwargs['gradient_steps'] = -1   # take gradient steps as much as possible after rollout
dqn_kwargs['tensorboard_log'] = './tb_log/'     # tensorboard log path
dqn_kwargs['verbose'] = 1
dqn_kwargs['seed'] = 32

dqn_kwargs['policy_kwargs'] = policy_kwargs

model = DQN(**dqn_kwargs)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

learning_kwargs = {'total_timesteps': int(1e5),
                   'callback': None,
                   'log_interval': 100,
                   'tb_log_name': 'run',
                   'eval_env': None,
                   'eval_freq': -1,
                   'n_eval_episodes': 5,
                   'eval_log_path': None,
                   'reset_num_timesteps': True}

eval_env = CubeEnv(3)       # eval env
learning_kwargs['total_timesteps'] = 10
learning_kwargs['log_interval'] = 1     # eval logging interval
learning_kwargs['eval_env'] = eval_env
learning_kwargs['n_eval_episodes'] = 3  # num of eval episodes
learning_kwargs['eval_log_path'] = './eval_log/'        # eval log path

model.learn(**learning_kwargs)