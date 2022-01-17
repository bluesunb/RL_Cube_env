import gym
import torch as th

from stable_baselines3.common.policies import BaseFeaturesExtractor


class CubeFeatureExtractor(BaseFeaturesExtractor):
    """
    1. Simplest
        (6*3*3, ) -> Linear(6*3*3, features_dim)
        (features_dim ,)

    2. Separated Linear
        (6*3*3, ) -> (6,9)
        (6,9) -> Linear(9, features_dim//6)
        (6, features_dim//6) -> (features_dim, )

    3. one-hot preprocessed obs to categorical
        (1, 324) -> (1,52,6)    (all cubic are one-hot encoded)
        (1,54,6) -> th.arange(6).T
        (1,54,1) -> (1,6,9)
        (1,6,9) -> Linear(9, features_dim//6)
        (1,6,features_dim//6) -> Flatten()
        (1,features_dim)
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 32*6):
        """
        Implementation of 3'rd structure
        """
        self.f_dim = features_dim//6
        self.cube_size_square = observation_space.sample().size//6
        super(CubeFeatureExtractor, self).__init__(observation_space, self.f_dim*6)
        self.lookup_table = th.unsqueeze(th.arange(6, dtype=th.float32), dim=0).T
        self.linear = th.nn.Linear(self.cube_size_square, self.f_dim)
        self.relu = th.nn.ReLU()
        self.flatten = th.nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        assert len(observations.shape) > 1, 'observation is not vectorized'

        device = observations.device
        self.lookup_table = self.lookup_table.to(device)
        self.linear= self.linear.to(device)
        self.relu = self.relu.to(device)
        self.flatten = self.flatten.to(device)

        n_obs= observations.shape[0]
        features = observations.view(n_obs, -1, 6)
        features = th.matmul(features, self.lookup_table)
        features = features.view(n_obs, 6, -1)
        features = self.relu(self.linear(features))
        return self.flatten(features)