import gym
import numpy as np
import matplotlib.pyplot as plt

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Sample.Cube.cube_env import CubeEnv

env = CubeEnv(3)

device = th.device('auto')
