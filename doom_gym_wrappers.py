
import gym
import numpy as np
import torch
from torchvision import transforms as T

class DoomObservation(gym.ObservationWrapper):
  def __init__(self, env, shape) -> None:
    super().__init__(env)
    self.shape = shape
    self.label_channel = env.label_channel
    obs_shape = (self.observation_space[0].shape[-1],) + self.shape
    obs_space_list = [
      gym.spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32), 
      self.observation_space[1]
    ]
    self.observation_space = gym.spaces.Tuple(obs_space_list)

  def permute_observation(self, observation):
    observation = np.transpose(observation, (2,0,1))
    observation = torch.tensor(observation.copy(), dtype=torch.float)
    return observation
  
  def observation(self, observation):
    obs = self.permute_observation(observation[0])
    obs = T.Resize(self.shape, T.InterpolationMode.NEAREST)(obs).squeeze(0) / 255.0
    
    # Make any relevant label masks completely white
    if self.label_channel >= 0:
      obs[self.label_channel,:,:] = (obs[self.label_channel,:,:] - 3.0/255.0).ceil()
    
    observation[0] = obs.numpy()
    return observation


class DoomMaxAndSkipEnv(gym.Wrapper):
  """
  Return only every ``skip``-th frame (frameskipping)

  :param env: the environment
  :param skip: number of ``skip``-th frame
  """

  def __init__(self, env: gym.Env, skip: int = 4):
    gym.Wrapper.__init__(self, env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = [None, None]
    self._tuple_obs = True
    self._skip = skip

  def step(self, action: int):
    """
    Step the environment with the given action
    Repeat action, sum reward, and max over last observations.

    :param action: the action
    :return: observation, reward, done, information
    """
    total_reward = 0.0
    done = None
  
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2:
        self._obs_buffer[0] = obs
      if i == self._skip - 1:
        self._obs_buffer[1] = obs
      total_reward += reward
      if done:
          break

    max_frame = [[] for _ in range(len(obs))]
    for i, obs in enumerate(self._obs_buffer):
      for j,obs_part in enumerate(obs):
        max_frame[j].append(obs_part)
        
    for i, obs_parts in enumerate(max_frame):
      max_frame[i] = np.max(obs_parts, axis=0)

    return max_frame, total_reward, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)
  
  
  
  # taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
      self.epsilon = epsilon
      self.shape = shape
      self.reset()
        
    def reset(self):
      self.mean = np.zeros(self.shape, "float64")
      self.var = np.ones(self.shape, "float64")
      self.count = self.epsilon
      
    def update(self, x):
      batch_mean = np.mean(x, axis=0)
      batch_var = np.var(x, axis=0)
      batch_count = x.shape[0]
      self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
      self.mean, self.var, self.count = update_mean_var_count_from_moments(
          self.mean, self.var, self.count, batch_mean, batch_var, batch_count
      )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
  delta = batch_mean - mean
  tot_count = count + batch_count

  new_mean = mean + delta * batch_count / tot_count
  m_a = var * count
  m_b = batch_var * batch_count
  M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
  new_var = M2 / tot_count
  new_count = tot_count

  return new_mean, new_var, new_count

  
class DoomNormalizeReward(gym.core.Wrapper):
  def __init__(self, env, epsilon=1e-8, clip=10):
    super(DoomNormalizeReward, self).__init__(env)
    self.num_envs = getattr(env, "num_envs", 1)
    self.is_vector_env = getattr(env, "is_vector_env", False)
    self.return_rms = RunningMeanStd(shape=())
    self.epsilon = epsilon
    self.clip = clip

  def step(self, action):
    obs, rews, dones, infos = self.env.step(action)
    if not self.is_vector_env:
      rews = np.array([rews])
    rews = np.clip(self.normalize(rews), -self.clip, self.clip)
    if not self.is_vector_env:
      rews = rews[0]
    return obs, rews, dones, infos

  def normalize(self, rews):
    self.return_rms.update(rews)
    return rews / np.sqrt(self.return_rms.var + self.epsilon)
  
  
  
class DoomNormalizeObservation(gym.core.Wrapper):
    def __init__(
        self,
        env,
        epsilon=1e-8,
    ):
        super(DoomNormalizeObservation, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space[0].shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space[0].shape)
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        if self.is_vector_env:
            obs_screen = self.normalize(obs[0])
        else:
            obs_screen = self.normalize(np.array([obs[0]]))[0]
        return (obs_screen,) + obs[1:], rews, dones, infos

    def reset(self):
        obs = self.env.reset()
        if self.is_vector_env:
            obs_screen = self.normalize(obs[0])
        else:
            obs_screen = self.normalize(np.array([obs[0]]))[0]
        return (obs_screen,) + obs[1:]

    def normalize(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)