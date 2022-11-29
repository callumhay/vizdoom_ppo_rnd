
from typing import List, Callable
import vizdoom as vzd
import numpy as np


class DoomReward(object):
  
  def __init__(self, reward_func) -> None:
    self.reward_func = reward_func
  
  def reinit(self, game: vzd.DoomGame) -> None:
    pass
  
  def update_and_calc_reward(self, game: vzd.DoomGame) -> float:
    reward = self.reward_func(game)
    return reward

class DoomRewardVar(object):
  
  def __init__(self, var_types: vzd.GameVariable|List[vzd.GameVariable], reward_func: Callable[[float,float],float]) -> None:
    self.var_types = var_types if isinstance(var_types, list) else [var_types]
    self.reward_func = reward_func
    self.curr_values = [0.0] * len(self.var_types)
    
  def reinit(self, game: vzd.DoomGame) -> None:
    for i, var in enumerate(self.var_types):
      self.curr_values[i] = game.get_game_variable(var)
    
  def update_and_calc_reward(self, game: vzd.DoomGame) -> float:
    reward = 0.0
    for i, var in enumerate(self.var_types):
      new_value = game.get_game_variable(var)
      if new_value != self.curr_values[i]:
        reward += self.reward_func(self.curr_values[i], new_value)
        self.curr_values[i] = new_value
    return reward

class DoomPosRewardVar(object):
  
  def __init__(self) -> None:
    self.curr_max_radius = 0.0
    self.init_xyz = np.array([0.,0.,0.])
    self.xyz_sum_vec = np.array([0.,0.,0.])
    self.steps_since_explore = 0
    
  def _curr_game_xyz(self, game: vzd.DoomGame) -> np.ndarray:
    return np.array([
      game.get_game_variable(vzd.GameVariable.POSITION_X),
      game.get_game_variable(vzd.GameVariable.POSITION_Y),
      game.get_game_variable(vzd.GameVariable.POSITION_Z),
    ])
  
  def reinit(self, game: vzd.DoomGame) -> None:
    self.curr_max_radius = 0.0
    self.init_xyz = self._curr_game_xyz(game)
    self.xyz_sum_vec = np.array([0.,0.,0.])
    self.steps_since_explore = 0

  def update_and_calc_reward(self, game: vzd.DoomGame) -> float:
    reward = 0.0
    curr_xyz = self._curr_game_xyz(game)
    diff_vec = curr_xyz - self.init_xyz
    self.xyz_sum_vec += diff_vec
    dist = np.sqrt(np.sum(diff_vec**2))
    
    radius_diff = dist - self.curr_max_radius
    if radius_diff > 0:
      reward += 0.1 * radius_diff
      self.curr_max_radius = dist
      self.steps_since_explore = 0
    else:
      # TODO: Punish for staying in the same area for a long time...
      self.steps_since_explore += 1

    return reward

class DoomAdvancedPosRewardVar(object):
  _SECTOR_SIZE = 50
  _EXPLORE_SECTOR_REWARD = 0.1
  _REVISIT_SECTOR_REWARD = 0#-0.1 # NOTE: Having a penalty here appears to keep convergence from happening? 
  _PUNISHMENT_INCREMENT_GAME_SECONDS_COEF = 15 # If an agent returns to a sector after this amount of elapsed game time, increment visits
  _REVISITS_INC_THRESHOLD = 3 # Number of times the agent has to revisit a sector before revisit reward
  
  def __init__(self) -> None:
    # Store _SECTOR_SIZEx_SECTOR_SIZE sectors that have been explored:
    # 1 if explored, 0 if not explored
    # [x // _SECTOR_SIZE][y // _SECTOR_SIZE][0|1]
    self.explored_map = {}
    self.curr_xyz = np.array([0.,0.,0.])
    
  def _curr_game_xyz(self, game: vzd.DoomGame) -> np.ndarray:
    return np.array([
      game.get_game_variable(vzd.GameVariable.POSITION_X),
      game.get_game_variable(vzd.GameVariable.POSITION_Y),
      game.get_game_variable(vzd.GameVariable.POSITION_Z),
    ])
  
  def reinit(self, game: vzd.DoomGame) -> None:
    self.explored_map = {}
    self.curr_xyz = self._curr_game_xyz(game)
    self.revisit_reward_frames = self._PUNISHMENT_INCREMENT_GAME_SECONDS_COEF * game.get_ticrate()
    
    
  def update_and_calc_reward(self, game: vzd.DoomGame) -> float:
    reward = 0.0
    
    start_xyz = self.curr_xyz
    end_xyz   = self._curr_game_xyz(game)
    self.curr_xyz = end_xyz
    
    # Use a variation on Bresenham's line algorithm to fill in the sectors we've visited
    # in the last step and accumulate the reward for each visited sector
    x0, x1 = int(start_xyz[0] // self._SECTOR_SIZE), int(end_xyz[0] // self._SECTOR_SIZE)
    y0, y1 = int(start_xyz[1] // self._SECTOR_SIZE), int(end_xyz[1] // self._SECTOR_SIZE)
    dx = np.abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -np.abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
      if x0 in self.explored_map:
        if y0 not in self.explored_map[x0]:
          self.explored_map[x0][y0] = [1, game.get_episode_time()]
          reward += self._EXPLORE_SECTOR_REWARD
        elif self._REVISIT_SECTOR_REWARD != 0:
          # If the agent continually revisits the same sector we punish them
          visits, prev_visit_step = self.explored_map[x0][y0]
          curr_steps = game.get_episode_time()
          if curr_steps-prev_visit_step >= self.revisit_reward_frames:
            visits += 1
            self.explored_map[x0][y0] = [visits, curr_steps]
            if visits > self._REVISITS_INC_THRESHOLD:
              reward += self._REVISIT_SECTOR_REWARD
      else:
        self.explored_map[x0] = {}
        self.explored_map[x0][y0] = [1, game.get_episode_time()]
        reward += self._EXPLORE_SECTOR_REWARD

      if x0 == x1 and y0 == y1: break
  
      e2 = 2*error
      if e2 >= dy:
        if x0 == x1: break
        error += dy
        x0 += sx
      if e2 <= dx:
        if y0 == y1: break
        error += dx
        y0 += sy
      
    return reward
