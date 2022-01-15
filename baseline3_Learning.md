# stable_baselines3 - Learning
## DQN
### Learning process
- `DQN.learn()` <br>
  <details>
    <summary>Code</summary>
  
  ```python
  from stable_baselines3.dqn.dqn import *
    
      def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,      # num of timesteps before logging
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,        # agent evaluate timestep freq
        n_eval_episodes: int = 5,   # num of episodes to evaluate agent
        tb_log_name: str = "DQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,   # (for logging) whether or not to reset timesteps
    ) -> OffPolicyAlgorithm:

        return super(DQN, self).learn( ... )
    ```
  
  </details>
    
    `OffPolicyAlgorithm.learn()` 으로 bound


- `OffPolicyAlgorithm.learn()`
  <details>
  <summary>Code</summary>
  
  ```python
    from stable_baselines3.common.off_policy_algorithm import *
  
      def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn( ... )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,     # model update freq (defined for each 'step', 'episode')
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )   # generate rollouts that length is train_freq 
                # 즉, train_freq마다 model update 해야하므로 맨 처음에는 train_freq 길이의 rollout를 쌓는다. 

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self
  ```
  
  </details>

  1. Collect rollouts for length train_freq
  2. if rollout is not ended, take train process

- `OffPolicyAlgorithm.collect_rollouts()`
  <details>
  <summary>Code</summary>

  ```python
  from stable_baselines3.common.off_policy_algorithm import *
  
  def collect_rollouts(
          self,
          env: VecEnv,
          callback: BaseCallback,
          train_freq: TrainFreq,
          replay_buffer: ReplayBuffer,
          action_noise: Optional[ActionNoise] = None,
          learning_starts: int = 0,
          log_interval: Optional[int] = None,
  ) -> RolloutReturn:
     
      self.policy.set_training_mode(False)
  
      episode_rewards, total_timesteps = [], []     # initialize episodic record
      num_collected_steps, num_collected_episodes = 0, 0
  
      # assertions
      ...
    
      # State-Dependent Exploration(sde), action noise initialize
      ...
  
      callback.on_rollout_start()
      continue_training = True
  
      while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
          done = False
          episode_reward, episode_timesteps = 0.0, 0
  
          while not done:
              # action reset when sde_sample_freq
              # So it can do better exploration for every sde_sample_freq
              if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                  # Sample a new noise matrix
                  self.actor.reset_noise()
              
              # action sample, take step
              action, buffer_action = self._sample_action(learning_starts, action_noise)
              new_obs, reward, done, infos = env.step(action)
  
              self.num_timesteps += 1
              episode_timesteps += 1
              num_collected_steps += 1
  
              # Give access to local variables
              callback.update_locals(locals())
              if callback.on_step() is False:
                  return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)
  
              episode_reward += reward
  
              # infos, done이 Wrapper로 감싸진 DummyVecEnv의 form (scalar -> [scalar]: 다중 env 지원을 위해 list화)
              # 으로 바꿔 update.
              self._update_info_buffer(infos, done)
  
              # Store data in replay buffer (normalized action and unnormalized observation)
              self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)
              
              # exploration이 progress_remaining이 일정 비율 이하가 되면 멈춘다.
              self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
              
              # DQN._on_step() : target_update_interval 마다 target_Q와 behavior_Q를 동기화
              # exploration_rate를 scheduler에 따라 update
              self._on_step()
              
              # num_collected_steps (단순히 rollout된 step 수) > train_freq 면 True(rollout 정지)
              if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                  break
  
          if done:  # episode 종료
              num_collected_episodes += 1   # episode 모은 개수 += 1
              self._episode_num += 1
              episode_rewards.append(episode_reward)    # episode record에 episode 전체 reward 저장
              total_timesteps.append(episode_timesteps)
  
              if action_noise is not None:
                  action_noise.reset()
  
              # Log training infos
              if log_interval is not None and self._episode_num % log_interval == 0:
                  self._dump_logs()
        
      # episode가 collect 되지 않고 rollout 종료 조건(num_collected_class > train_freq)에 도달하면 0 (즉 무소용)
      # episide가 하나 이상 모였다면 episode_rewards가 의미 있으므로 (1개 이상 저장돼 있으므로) mean return
      mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0
  
      callback.on_rollout_end()
  
      return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)
   ```
  
  </details>

  1. episode가 끝나지 않았으면, mean_reward가 0인 RolloutReturn을 반환
  2. episode가 끝났으면, RolloutReturn을 반환


- `DQN.learn() ->`  
  > `OffPolicyAlgorithm.learn()` 으로 바운드 한다.

- `OffPolicyAlgorithm.learn() ->`  
  > `collect_rollouts()` + `train()`

- `OffPolicyAlgorithm.collect_rollouts()` ->  
  > `train_freq`만큼의 rollout 수집

- `OffPolicyAlgorithm.train() ->`  
  > `DQN.train()`으로 바운드 

- `DQN.train() ->`  
  > `QNetwork` 의 td-error를 구해 optimizer gradient 수행

- `QNetwork.forward() ->`  
  > `net_arch`에 의해서 생성된 mlp q_net에 대해 `feature_extractor`로 처리된 obs를 forwarding.

- `QNetwork.create_mlp() ->`  
  > `'net_arch`를 hidden layer dims로 가지며 `features_dim`, `action_space.n`를 각각 input_dim, output_dim으로 가지는 mlp 생성
  
- `FeaturesExtractor.forward() ->`  
  > `features_extractor_class`를 통해 obs를 처리

 
### Learning Parameters

#### DQN parameters

- `policy`  
  - `Union[str, Type[DQNPolicy]]`  
  model이 사용할 policy (`MlpPolicy`, `CnnPolicy` ,...)

- `env`
  - `Union[GymEnv, str]`
  learning의 대상이 되는 environment

- `learning_rate`
  - `Union[float, Schedule] = 1e-4`  
  learning rate

- `buffer_size`
  - `int = 1e6`  
  replay_buffer의 크기

- `learning_starts`
  - `int = 5e4`  
  learning 시작 전에 모을 transition 의 개수

- `batch_size`
  - `Optional[int] = 32`  
  각 gradient 업데이트시에 필요한 minibatch 크기

- `tau`
  - `float = 1.0`  
  soft update(Polyak update)에 쓰이는 coefficient

- `gamma`
  - `float = 0.99`  
  discount factor

- `train_freq`
  - `Union[int, Tuple[int, str] = 4`
  model의 update 주기 (not 횟수)
  - If `int`: 모델의 업데이트 timestep 주기
  - If `Tuple[int, str]`: `(5, 'step')`의 경우, `step`가 5 지날 때 마다 update 
    `(4, 'episode')`의 경우 `episode`가 4 지날 때 마다 update  
  
  이 때 `train_freq` 마다 모델을 업데이트 하므로 `train_freq` 만큼의 rollout을 buffer에 저장하게 되고, `train_freq` 마다 샘플링된다.
  
  <details>
  <summary>rollout</summary>
  여러개의 transition을 모아놓은 것.  
  collect rollout : collect transitions with length of `train_freq`
  </details>  

- `gradient_steps`
  - `int = 1`
  rollout (collecting transitions) 이후 수행할 gradient step의 횟수.  
  => 즉, model의 업데이트 횟수 (not 주기)  
  - if `gradient_steps` == -1: 할 수 있는 만큼(= 모은 rollout 갯수) 수행

- `replay_buffer_class`
  - `Optional[ReplayBuffer] = None`  
  `HerReplayBuffer`의 instance. `None`일 때 자동선택

- `replay_buffer_kwargs`  
  See `ReplayBuffer` class.

- `optimize_memory_usage`
  - `bool = False`  
  complexity를 늘려 replay buffer에서 메모리 효율성을 높임

- `target_update_interval`  
  - `int = 1e4`
  target network의 update interval. (env steps로 계산)

- `exploration_fraction`
  - `float = 0.1`  
  전체 training period에 대해 exploration_rate가 줄어드는 steps의 비율

- `exploration_initial_eps`
  - `float = 1.0`  
  initial $\epsilon$-greedy '$\epsilon$'

- `exploration_final_eps`
  - `float = 0.05`  
  final $\epsilon$-greedy '$\epsilon$'

- `max_grad_norm`
  - `float = 10.0`  
  gradient clipping 에서의 최대값

- `tensorboard_log`
  - `Optional[str] = None`  
  tensorboard log location. If `None`, no logging.

- `create_eval_env`
  - `bool = False`  
  agent를 주기적으로 evaluate 하기 위한 2차 env를 만들 것인지 여부.  
  string으로 된 env만 가능 (= repo에 등록된 env만 가능)

- `policy_kwargs`
  `DQNPolicy` 참고.

- `verbose`
  - `int = 0`  
  0: no input  
  1: info  
  2: debug

- `seed`
  - `Optional[int] = None`  
  random seed

- `device`
  - `Union[device, str] = "auto"`  
  one of 'cpu, cuda, auto'

- `_init_setup_model`
  - `bool = True`  
  `DQN` 인스턴스가 생성될 때 network들도 만들 것인지 여부
  
#### Learning parameters
- `total_timesteps`
  - `int`  
  train에 쓰일 총 env step의 수

- `callback`
  - `MaybeCallback = None`  
  every step 마다 호출되는 callback

- `log_interval`
  - `int = 100`  
  각 logging 사이의 timestep 간격
  
- `tb_log_name`
  - `str = "run"`  
  TensorBoard log의 run의 이름

- `eval_env`
  - `Optional[GymEnv] = None`  
  agent를 evaluate 하기 위한 env

- `eval_freq`
  - `int = -1`  
  agent evaluation interval

- `n_eval_episodes`
  - `int = 5`  
  agent evaluation의 episode 수

- `eval_log_path`
  - `Optional[str] = None`  
  evaluation log의 path

- `reset_num_timesteps`
  - `bool = True`  
  log 기록에서 현재 timestep을 reset 할지 여부

#### Trainable parameters

<details>
<summary>QNetwork</summary>

`QNetwork.q_net`은 `create_mlp`로부터 mlp layer로 만들어진다.  
`QNetwork.q_net`은 `net_arch`를 통해 layer dim, depth가 결정된다.

</details>

<details>
<summary>FeaturesExtractor</summary>

`FeaturesExtractor` 내에는 feature를 잡기 위한 trainable layer가 있다.  
`FeaturesExtractor`의 layer 구조는 `features_extractor_kwargs`를 통해 tuning 가능하다.

</details>