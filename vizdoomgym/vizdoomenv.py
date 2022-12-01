import os
import warnings
import itertools
from typing import List

import gym
from gym import spaces
import vizdoom as vzd
import numpy as np

import torch
from torchvision import transforms as T

from doom_general_env_config import DoomGeneralEnvConfig

turn_off_rendering = False
try:
    from gym.envs.classic_control import rendering
except Exception as e:
    print(e)
    turn_off_rendering = True

CONFIGS = [
    ["basic.cfg"],  # 0
    ["deadly_corridor.cfg"],  # 1
    ["defend_the_center.cfg"],  # 2
    ["defend_the_line.cfg"],  # 3
    ["health_gathering.cfg"],  # 4
    ["my_way_home.cfg"],  # 5
    ["predict_position.cfg"],  # 6
    ["take_cover.cfg"],  # 7
    ["deathmatch.cfg"],  # 8
    ["health_gathering_supreme.cfg"],  # 9
    ["basic_more_actions.cfg"] # 10
]

class VizdoomEnv(gym.Env):
    def __init__(self, level, **kwargs):

      # Parse keyword arguments
      self.depth = kwargs.get("depth", False)
      self.labels = kwargs.get("labels", False)
      self.automap = kwargs.get("automap", False)
      self.frame_skip = kwargs.get("frame_skip", 1) # Use the MaxAndSkipEnv Wrapper instead for skipping frames!
      self.smart_actions = kwargs.get("smart_actions", False)
      self.always_run = kwargs.get("always_run", False)
      self.custom_config = kwargs.get("custom_config", None)
      
      window_visible = kwargs.get("window_visible", False)
      game_dir = kwargs.get("game_dir", "bin")
      wad_gamepath = kwargs.get("wad_path", os.path.join(game_dir, "doom2.wad"))
      resolution = kwargs.get("resolution", vzd.ScreenResolution.RES_640X480)
      max_buttons_pressed = 2 if self.smart_actions else kwargs.get("max_buttons_pressed", 1)

      # init game
      self.game = vzd.DoomGame()
      self.game.set_doom_game_path(wad_gamepath)
      
      if level >= 0 and level < len(CONFIGS):
        self.game.load_config(os.path.join(game_dir, CONFIGS[level][0]))
        
      self.game.set_screen_resolution(resolution)
      self.game.set_episode_start_time(10)
      self.game.set_window_visible(window_visible)
      self.game.set_depth_buffer_enabled(self.depth)
      self.game.set_labels_buffer_enabled(self.labels)
      self.game.set_automap_buffer_enabled(self.automap)
      #self.game.set_console_enabled(True)
      if self.automap:
        #self.game.set_render_hud(False)
        self.game.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)
        # Map's colors can be changed using CVARs
        # Full list is available here: https://zdoom.org/wiki/CVARs:Automap#am_backcolor
        self.game.add_game_args("+am_followplayer 1")
        self.game.add_game_args("+viz_am_scale 1.25")
        self.game.add_game_args("+am_overlay 1")
        self.game.add_game_args("+am_backcolor 000000")

        
      if self.custom_config is not None:
        self.custom_config.load(self)
      
      self.game.init()
      self.state = None
      self.viewer = None

      delta_buttons, binary_buttons = self.__parse_available_buttons()
      # check for valid max_buttons_pressed
      if max_buttons_pressed > self.num_binary_buttons > 0:
        warnings.warn(
            f"max_buttons_pressed={max_buttons_pressed} "
            f"> number of binary buttons defined={self.num_binary_buttons}. "
            f"Clipping max_buttons_pressed to {self.num_binary_buttons}.")
        max_buttons_pressed = self.num_binary_buttons
      elif max_buttons_pressed < 0:
        raise RuntimeError(f"max_buttons_pressed={max_buttons_pressed} < 0. Should be >= 0. ")
    
      # Specify the action space(s)
      self.max_buttons_pressed = max_buttons_pressed
      self.action_space = self.__get_action_space(delta_buttons, binary_buttons)

      # specify observation space(s)
      num_channels = self.game.get_screen_channels()

      if self.depth:
        self.depth_channel = num_channels
        num_channels += 1
      else:
        self.depth_channel = -1

      if self.labels:
        self.label_channel = num_channels
        num_channels += 1 
      else: 
        self.label_channel = -1
      
      if self.automap:
        self.automap_channel = num_channels
        num_channels += 1
      else:
        self.automap_channel = -1
        
      list_spaces: List[gym.Space] = [
        spaces.Box(0, 255, (
            self.game.get_screen_height(),
            self.game.get_screen_width(),
            num_channels,
          ), dtype=np.uint8)
      ]
      if self.custom_config is not None:
        list_spaces += self.custom_config.game_variable_spaces()
      else:
        list_spaces.append(gym.spaces.Box(0.0, 1.0, (7,)))
        self.game.add_available_game_variable(vzd.GameVariable.POSITION_X)
        self.game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
        self.game.add_available_game_variable(vzd.GameVariable.ANGLE) 
        self.game.add_available_game_variable(vzd.GameVariable.HEALTH) 
        self.game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)

      self.observation_space = spaces.Tuple(list_spaces)


    def __parse_binary_buttons(self, env_action, agent_action):
      if self.num_binary_buttons != 0:
        if self.num_delta_buttons != 0:
          agent_action = agent_action["binary"]

        if isinstance(agent_action, int) or (agent_action.size is not None and agent_action.size == 1):
          agent_action = self.button_map[agent_action]

        # binary actions offset by number of delta buttons
        env_action[self.num_delta_buttons:] = agent_action

    def __parse_delta_buttons(self, env_action, agent_action):
      if self.num_delta_buttons != 0:
        if self.num_binary_buttons != 0:
          agent_action = agent_action["continuous"]
        # delta buttons have a direct mapping since they're reorganized to be prior to any binary buttons
        env_action[0:self.num_delta_buttons] = agent_action

    def __build_env_action(self, agent_action):
      # encode users action as environment action
      env_action = np.array([0 for _ in range(self.num_delta_buttons + self.num_binary_buttons)], dtype=np.float32)
      self.__parse_delta_buttons(env_action, agent_action)
      self.__parse_binary_buttons(env_action, agent_action)
      return env_action

    def is_episode_finished(self):
      return self.game.get_state() == None or self.game.is_episode_finished()
    
    def is_map_ended(self):
      return self.game.is_episode_finished() and not self.game.is_player_dead() and self.game.get_episode_time() < self.game.get_episode_timeout()

    def seed(self, seed:int):
      self.game.set_seed(seed)

    def step(self, action):
      assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
      assert self.state is not None, "Call `reset` before using `step` method."
      
      act = self.__build_env_action(action)
      try:
        reward = self.game.make_action(act, self.frame_skip)
      except:
        print("Vizdoom window was closed, exiting.")
        exit(0)
      
      if self.custom_config is not None:
        reward += self.custom_config.step_reward(self.game)
      
      self.state = self.game.get_state()
      done = self.game.is_episode_finished()
      info = {"dummy": 0.0}
      #if done:
      #  info["map_complete"] = 1 if self.is_map_ended() else 0 

      return self.__collect_observations(), reward, done, info

    def reset(self):
      try:
        self.game.new_episode()
        self.state = self.game.get_state()
        if self.custom_config is not None:
          self.custom_config.reset(self.game)
      except:
        print("Vizdoom window was closed, exiting.")
        exit(0)
        
      return self.__collect_observations()

    def __collect_observations(self):
      observations = []
      
      if self.state is not None:
        observation = np.transpose(self.state.screen_buffer, (1, 2, 0))
        
        if self.depth:
          cat_axis = self.state.depth_buffer.ndim
          depth_observation = np.expand_dims(self.state.depth_buffer, cat_axis)
          observation = np.concatenate([observation, depth_observation], cat_axis)
          
        if self.labels:
          cat_axis = self.state.labels_buffer.ndim
          labels_observation = np.expand_dims(self.state.labels_buffer, cat_axis)#np.expand_dims((self.state.labels_buffer>3)*255, cat_axis)
          observation = np.concatenate([observation, labels_observation], cat_axis)
        
        if self.automap:
          automap_obs = torch.tensor(self.state.automap_buffer, dtype=torch.float)
          automap_obs = (T.Grayscale()(automap_obs)).numpy()
          automap_obs = np.transpose(automap_obs, (1, 2, 0))
          observation = np.concatenate([observation, automap_obs], 2)
        
        observations.append(observation)
        observations += DoomGeneralEnvConfig.game_variable_observations(self.game)

      else:
        # There is no state in the terminal step, so a "zero observation is returned instead"
        for space in self.observation_space:
          observations.append(np.zeros(space.shape, dtype=space.dtype))

      return observations

    def render(self, mode="human"):
      if turn_off_rendering:
        return
      try:
        img = self.game.get_state().screen_buffer
        img = np.transpose(img, [1, 2, 0])

        if self.viewer is None:
          self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
      except AttributeError:
        pass
        
    def close(self):
      super().close()
      if self.viewer:
        self.viewer.close()
      if self.game:
        self.game.close()

    @staticmethod
    def get_keys_to_action():
      # you can press only one key at a time!
      keys = {
        (): 2,
        (ord("a"),): 0,
        (ord("d"),): 1,
        (ord("w"),): 3,
        (ord("s"),): 4,
        (ord("q"),): 5,
        (ord("e"),): 6,
      }
      return keys


    def __get_action_space(self, delta_buttons, binary_buttons):
      """
      return action space:
          if both binary and delta buttons defined in the config file, action space will be:
            Dict("binary": MultiDiscrete|Discrete, "continuous", Box)
          else:
            action space will be only one of the following MultiDiscrete|Discrete|Box
      """
      if self.num_delta_buttons == 0:
        return self.__get_binary_action_space(binary_buttons)
      elif self.num_binary_buttons == 0:
        return self.__get_continuous_action_space(delta_buttons)
      else:
        return gym.spaces.Dict({
          "binary": self.__get_binary_action_space(),
          "continuous": self.__get_continuous_action_space()
        })

    def __get_binary_action_space(self, binary_buttons):
      """
      return binary action space: either Discrete(n)/MultiDiscrete([2,]*num_binary_buttons)
      """
      if self.max_buttons_pressed == 0:
        button_space = gym.spaces.MultiDiscrete([2,] * self.num_binary_buttons)
      else:
        if self.smart_actions:
          button_dict = {button: idx for idx, button in enumerate(binary_buttons)}
          do_strafe_check = vzd.Button.MOVE_LEFT in button_dict and vzd.Button.MOVE_RIGHT in button_dict
          do_turn_check   = vzd.Button.TURN_LEFT in button_dict and vzd.Button.TURN_RIGHT in button_dict
          do_fwbk_check   = vzd.Button.MOVE_FORWARD in button_dict and vzd.Button.MOVE_BACKWARD in button_dict
          do_atkuse_check = vzd.Button.ATTACK in button_dict and vzd.Button.USE in button_dict
          do_speed_check  = vzd.Button.SPEED in button_dict
          self.button_map = []
          for action in itertools.product((0, 1), repeat=self.num_binary_buttons):
            
            if (self.max_buttons_pressed >= sum(action) >= 0):
              # You don't turn/move left and turn right simulateously...
              if do_strafe_check and action[button_dict[vzd.Button.MOVE_LEFT]] == 1 and action[button_dict[vzd.Button.MOVE_RIGHT]] == 1:
                continue
              elif do_turn_check and action[button_dict[vzd.Button.TURN_LEFT]] == 1 and action[button_dict[vzd.Button.TURN_RIGHT]] == 1:
                continue
              # You don't move forward/backward simultaneously...
              elif do_fwbk_check and action[button_dict[vzd.Button.MOVE_FORWARD]] == 1 and action[button_dict[vzd.Button.MOVE_BACKWARD]] == 1:
                continue
              # You don't shoot and use simultaneously...
              elif do_atkuse_check and action[button_dict[vzd.Button.ATTACK]] == 1 and action[button_dict[vzd.Button.USE]] == 1:
                continue
              
              
              elif do_speed_check:
                if action[button_dict[vzd.Button.SPEED]] == 1:
                  # speed is not a standalone action...
                  if sum(action) == 1: continue 
                  # speed must be used in combination with actual movement...
                  if self.always_run or (do_strafe_check and action[button_dict[vzd.Button.MOVE_LEFT]] != 1 and action[button_dict[vzd.Button.MOVE_RIGHT]] != 1) \
                    and (do_fwbk_check and action[button_dict[vzd.Button.MOVE_FORWARD]] != 1 and action[button_dict[vzd.Button.MOVE_BACKWARD]] != 1) \
                    and (do_turn_check and action[button_dict[vzd.Button.TURN_LEFT]] != 1 and action[button_dict[vzd.Button.TURN_RIGHT]] != 1):
                    continue
              
              action_list = list(action)
              # Make sure any action that involves movement also has speed activated when always_run is on
              if self.always_run and do_speed_check:
                is_strafing = do_strafe_check and (action[button_dict[vzd.Button.MOVE_LEFT]] == 1 or action[button_dict[vzd.Button.MOVE_RIGHT]] == 1)
                is_movingfb = do_fwbk_check and (action[button_dict[vzd.Button.MOVE_FORWARD]] == 1 or action[button_dict[vzd.Button.MOVE_BACKWARD]] == 1)
                is_turning  = do_turn_check and (action[button_dict[vzd.Button.TURN_LEFT]] == 1 or action[button_dict[vzd.Button.TURN_RIGHT]] == 1)
                if is_strafing or is_movingfb or is_turning:
                  action_list[button_dict[vzd.Button.SPEED]] = 1
              
              self.button_map.append(np.array(action_list))
          assert len(np.unique(self.button_map, axis=1)) == len(self.button_map)
          self.button_map = self.button_map[1:] # Remove the no-action/noop row
        else:
          self.button_map = [
            np.array(list(action)) for action in itertools.product((0, 1), repeat=self.num_binary_buttons)
            if (self.max_buttons_pressed >= sum(action) >= 0)
          ]
          
        button_space = gym.spaces.Discrete(len(self.button_map))
      return button_space

    def __get_continuous_action_space(self, delta_buttons):
      """
      return continuous action space: Box(float32.min, float32.max, (num_delta_buttons,), float32)
      """
      return gym.spaces.Box(
        np.finfo(np.float32).min, 
        np.finfo(np.float32).max, 
        (self.num_delta_buttons,), 
        dtype=np.float32
      )

    def __parse_available_buttons(self):
      """
      Parses the currently available game buttons,
      reorganizes all delta buttons to be prior to any binary buttons
      sets self.num_delta_buttons, self.num_binary_buttons
      """
      delta_buttons = []
      binary_buttons = []
      for button in self.game.get_available_buttons():
        if vzd.is_delta_button(button) and button not in delta_buttons:
            delta_buttons.append(button)
        else:
            binary_buttons.append(button)
      # force all delta buttons to be first before any binary buttons
      self.game.set_available_buttons(delta_buttons + binary_buttons)
      self.num_delta_buttons = len(delta_buttons)
      self.num_binary_buttons = len(binary_buttons)
      if delta_buttons == binary_buttons == 0:
        raise RuntimeError("No game buttons defined. Must specify game buttons using `available_buttons` in the config file.")
      return delta_buttons, binary_buttons