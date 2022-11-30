import argparse
import os
import random
import time
import re
from distutils.util import strtobool

import gym
import vizdoom as vzd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from torch.utils.tensorboard import SummaryWriter

from net_utils import layer_init_ortho, keycard_pixels_from_obs, keycard_rect
from rnd_network import RNDNetwork
from sd_encoder import SDEncoder
from doom_gym_wrappers import register_vizdoom_gym_envs, DoomMaxAndSkipEnv, DoomObservation #, DoomNormalizeReward
from doom_general_env_config import DoomGeneralEnvConfig, NUM_POSITION_GAMEVARS, POSITION_NUM_VALUES, ANGLE_HEALTH_AMMO_NUM_VALUES

register_vizdoom_gym_envs()

class RunningMeanStd(object):
  def __init__(self, epsilon=1e-4, shape=()):
    self.epsilon = epsilon
    self.shape = shape
    self.reset()

  def reset(self):
    self.mean = np.zeros(self.shape, dtype=np.float64)
    self.var = np.ones(self.shape, dtype=np.float64)
    self.count = self.epsilon

  def update(self, x):
    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)
    batch_count = x.shape[0]
    self.update_from_moments(batch_mean, batch_var, batch_count)

  def update_from_moments(self, batch_mean, batch_var, batch_count):
    self.mean, self.var, self.count = RunningMeanStd._update_mean_var_count_from_moments(
        self.mean, self.var, self.count, batch_mean, batch_var, batch_count
    )
    
  def _update_mean_var_count_from_moments(
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
        
class SimpleNormalizeReward(object):
  def __init__(self, num_envs, epsilon=1e-8):
    super(SimpleNormalizeReward, self).__init__()
    self.return_rms = RunningMeanStd(shape=(num_envs,))
    self.epsilon = epsilon

  def normalize(self, rewards):
    self.return_rms.update(np.expand_dims(rewards, axis=0))
    return rewards / np.sqrt(self.return_rms.var + self.epsilon)

  def reset(self):
    self.return_rms.reset()

def parse_args():
  bool_val_fn = lambda x: bool(strtobool(x.strip()))
  
  parser = argparse.ArgumentParser()
  # Top-level program arguments
  parser.add_argument("--model", type=str, default="", help="Preexisting model to load (.chkpt file)")
  parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
  parser.add_argument("--gym-id", type=str, default="VizdoomDoomGame-v0", help="the id of the gym environment") # See register_vizdoom_gym_envs() for a listing of all possible ids
  parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
  parser.add_argument("--seed", type=int, default=42, help="RNG seed of the experiment")
  parser.add_argument("--total-timesteps", type=int, default=100000000, help="total timesteps of the experiments")
  parser.add_argument("--save-timesteps", type=int, default=50000, help="Timesteps between network saves")
  parser.add_argument("--torch-deterministic", type=bool_val_fn, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
  parser.add_argument("--cuda", type=bool_val_fn, default=True, help="if toggled, cuda will be enabled by default")
  parser.add_argument("--track", type=bool_val_fn, default=False, help="if toggled, this experiment will be tracked with Weights and Biases")
  parser.add_argument("--wandb-project-name", type=str, default="vizdoom-ppo-rnd", help="the wandb's project name")
  parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
  parser.add_argument("--capture-video", type=bool_val_fn, default=False, help="weather to capture videos of the agent performances (check out `videos` folder)")

  # Game specific arguments
  parser.add_argument("--gamepath", type=str, default="bin", help="The path to where the doom.wad and doom2.wad files can be found")
  parser.add_argument("--map", type=str, default="MAP01", help="The map to load when playing the regular doom game scenario")
  parser.add_argument("--multidiscrete-actions", type=bool_val_fn, default=False, 
    help="Whether the agent uses multidiscrete actions (up to 2 at a time) or not - this is disabled automatically if smart-actions are enabled") # NOTE: Multidiscrete converges faster!
  parser.add_argument("--smart-actions", type=bool_val_fn, default=True,
    help="Whether to use smart multdiscrete actions (e.g., you don't move left and right simulataneously), this limits actions to 2 at a time")
  parser.add_argument("--always-run", type=bool_val_fn, default=True, help="Whether all move actions are always done with speed/run")
  parser.add_argument("--automap", type=bool_val_fn, default=False, help="Whether to use the automap buffer as part of the observations")
  parser.add_argument("--labels", type=bool_val_fn, default=True, help="Whether to use the labels buffer as part of the observations")
  parser.add_argument("--depth", type=bool_val_fn, default=False, help="Whether to use the depth buffer as part of the observations")
  
  # Network specific arguments
  parser.add_argument("--z-channels", type=int, default=48, help="Number of z-channels on the last layer of the convolutional network")
  parser.add_argument("--net-output-size", type=int, default=4608, help="Output size of the convolutional network, input size to the LSTM")
  parser.add_argument("--lstm-hidden-size", type=int, default=1468, help="Hidden size of the LSTM")
  parser.add_argument("--lstm-num-layers", type=int, default=1, help="Number of layers in the LSTM") # NOTE: More than one layer doesn't appear to have much benefit (and slows training down a lot!)
  parser.add_argument("--lstm-dropout", type=float, default=0.0, help="Dropout fraction [0,1] in the LSTM")
  parser.add_argument("--obs-shape", type=str, default="60,80", # NOTE: 60,80 works well, increasing to 69,92 doesn't appear to help convergence...
    help="Shape of the RGB screenbuffer (height, width) after being processed (when fed to the convolutional network).")
  parser.add_argument("--ch-mult", type=str, default="1,2,3,4", 
    help="Multipliers of '--starting-channels', for the number of channels for each layer of the convolutional network")
  parser.add_argument("--num-res-blocks", type=int, default=1, help="Number of ResNet blocks in the convolutional network")
  parser.add_argument("--starting-channels", type=int, default=32, help="Initial number of channels in the convolutional network")
  parser.add_argument("--rnd-output-size", type=int, default=512, help="Output size of the predictor and target networks")
  
  # Algorithm specific arguments
  parser.add_argument("--num-explore-steps", type=int, default=1000, help="the number of pre-training steps to initialize the normalizers for rewards")
  parser.add_argument("--num-envs", type=int, default=20, help="the number of parallel game environments")
  parser.add_argument("--num-steps", type=int, default=256, help="the number of steps to run in each environment per policy rollout")
  parser.add_argument("--anneal-lr", type=bool_val_fn, default=True, help="Toggle learning rate annealing for policy and value networks")
  parser.add_argument("--reward-i-coeff", type=float, default=0.01, help="Coefficient for the intrinsic reward to balance it with the extrinsic reward")
  parser.add_argument("--gamma-e", type=float, default=0.999, help="the discount factor gamma for extrinsic rewards")
  parser.add_argument("--gamma-i", type=float, default=0.99, help="the discount factor gamma for intrinsic rewards")
  parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
  parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
  parser.add_argument("--update-epochs", type=int, default=4, help="the K epochs to update the policy")
  parser.add_argument("--norm-adv", type=bool_val_fn, default=True, help="Toggles advantages normalization")
  parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
  parser.add_argument("--ent-coef", type=float, default=0.009, help="coefficient of the entropy")
  parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
  parser.add_argument("--pt-coef", type=float, default=0.005, help="coefficient of the predictor-target loss function")
  parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
  parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
  
  args = parser.parse_args()
  
  # Multi-discrete and smart actions are mutually exclusive... 
  # TODO: Should probably make them an enum at some point.
  if args.smart_actions:
    args.multidiscrete_actions = False
  elif args.multidiscrete_actions:
    args.smart_actions = False
  
  args.obs_shape = tuple([int(item) for item in args.obs_shape.split(',')])
  args.ch_mult = [int(item) for item in args.ch_mult.split(',')]
  args.batch_size = int(args.num_envs * args.num_steps)
  args.minibatch_size = int(args.batch_size // args.num_minibatches)

  return args


def make_env(args, seed, idx, run_name):
  def thunk():
    max_buttons_pressed = 0 if args.multidiscrete_actions else 1
    doom_game_config = DoomGeneralEnvConfig(args.map, disable_explore_reward=True, living_reward=-0.005) if args.gym_id == "VizdoomDoomGame-v0" else None
    wad_file = "doom2.wad" if re.search(r"[Ee]\d+[Mm]\d+", args.map) == None else "doom.wad" # Doom maps are formatted "E1M1", whereas Doom2 maps are "Map01"
    game_dir = os.path.join(os.path.dirname(__file__), args.gamepath)
    env = gym.make(
      args.gym_id,
      set_window_visible=(idx==0),
      game_dir=game_dir,
      wad_path=os.path.join(game_dir, wad_file),
      resolution=vzd.ScreenResolution.RES_200X150,
      max_buttons_pressed=max_buttons_pressed,
      smart_actions=args.smart_actions,
      labels=args.labels,
      depth=args.depth,
      automap=args.automap,
      custom_config=doom_game_config,
      always_run=args.always_run,
    )
    
    env = DoomObservation(env, shape=args.obs_shape)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if args.capture_video and idx == 0:
      env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = DoomMaxAndSkipEnv(env, skip=4)
    
    # NOTE: Removing reward normalization for just extrinsic reward is REQUIRED to get the agent to learn properly
    #env = DoomNormalizeReward(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

  return thunk


class Agent(nn.Module):
  def __init__(self, args, envs):
    super(Agent, self).__init__()
    
    net_output_size  = args.net_output_size
    lstm_hidden_size = args.lstm_hidden_size
    
    self.pixel_convnet = SDEncoder(args, envs)
    
    # NOTE: I've only tested with values of 256 for positional encoding and 64 for the other game vars
    POS_EMBED_SIZE = 256
    GAME_VAR_EMBED_SIZE = 64
    
    self.num_game_and_pos_vars = envs.single_observation_space[1].shape[0]
    self.pos_embeddings = nn.ModuleList([nn.Embedding(POSITION_NUM_VALUES, POS_EMBED_SIZE) for _ in range(NUM_POSITION_GAMEVARS)])
    self.other_var_embeddings = nn.ModuleList([
      nn.Embedding(ANGLE_HEALTH_AMMO_NUM_VALUES, GAME_VAR_EMBED_SIZE) for _ in range(self.num_game_and_pos_vars-NUM_POSITION_GAMEVARS)
    ])
    
    # Keycard pixel crop: This is used to provide the keycard pixels directly to the networks of the algorithm so that
    # it can learn to explore and change its behaviour based on whether keycards are present or not in the in-game HUD
    kc_rect = keycard_rect(envs.single_observation_space[0].shape)
    kc_w = kc_rect[2]-kc_rect[0]
    kc_h = kc_rect[3]-kc_rect[1]
    
    self.game_var_size = NUM_POSITION_GAMEVARS*POS_EMBED_SIZE + (self.num_game_and_pos_vars-NUM_POSITION_GAMEVARS)*GAME_VAR_EMBED_SIZE
    lstm_input_size = net_output_size + self.game_var_size + (3*kc_w*kc_h) # NOTE: 3 is for the r, g, and b channels
    
    self.predictor_rnd_net = RNDNetwork(envs.single_observation_space[0].shape, args.rnd_output_size, is_predictor=True)
    self.target_rnd_net = RNDNetwork(envs.single_observation_space[0].shape, args.rnd_output_size, is_predictor=False)
    # Freeze the target network
    self.target_rnd_net.eval()
    for p in self.target_rnd_net.parameters():
      p.requires_grad = False
    
    # NOTE: LSTM appears to like a 4:1 ratio of input size to hidden size, increasing
    # this ratio is detrimental to the network (doesn't converge or takes a very long time)
    self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, dropout=args.lstm_dropout, num_layers=args.lstm_num_layers)
    for name, param in self.lstm.named_parameters():
      if "bias" in name:
        nn.init.constant_(param, 0)
      elif "weight" in name:
        nn.init.orthogonal_(param, 1.0)
    
    self.multidiscrete_actions = args.multidiscrete_actions
    if self.multidiscrete_actions:
      self.nvec = envs.single_action_space.nvec
      action_space_size = self.nvec.sum()
    else:
      action_space_size = envs.single_action_space.n
             
    self.actor  = layer_init_ortho(nn.Linear(lstm_hidden_size, action_space_size), std=0.01)
    self.critic = layer_init_ortho(nn.Linear(lstm_hidden_size, 2), std=1) # Two value heads (intrinsic and extrinsic rewards)

  def get_states(self, visual_obs, gamevars_obs, lstm_state, done):
    pixel_conv_out = self.pixel_convnet(visual_obs)
    
    var_tuples = torch.chunk(gamevars_obs, self.num_game_and_pos_vars, -1)
    # The last 4 tuples are the position information
    embed_pos = torch.cat([self.pos_embeddings[i](t.int()).squeeze(1) for i,t in enumerate(var_tuples[-NUM_POSITION_GAMEVARS:])], -1)
    # The first 3 tuples are player orientation, health, and ammo
    embed_other_vars = torch.cat([self.other_var_embeddings[i](t.int()).squeeze(1) for i,t in enumerate(var_tuples[:-NUM_POSITION_GAMEVARS])], -1)
    # The keycard pixel c, h, and w values are all flattened for concatenation
    keycard_pixels = keycard_pixels_from_obs(visual_obs).flatten(1)
    
    hidden = torch.cat([pixel_conv_out, embed_pos, embed_other_vars, keycard_pixels], -1) # Put it all together for input to LSTM
    
    # LSTM logic
    batch_size = lstm_state[0].shape[1]
    hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
    done = done.reshape((-1, batch_size))
    new_hidden = []
    for h, d in zip(hidden, done):
      h, lstm_state = self.lstm(
        h.unsqueeze(0),
        ((1.0 - d).view(1, -1, 1) * lstm_state[0], (1.0 - d).view(1, -1, 1) * lstm_state[1]),
      )
      new_hidden += [h]
    new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
    return new_hidden, lstm_state

  def get_value(self, visual_obs, gamevars_obs, lstm_state, done):
    hidden, _ = self.get_states(visual_obs, gamevars_obs, lstm_state, done)
    return self.critic(hidden)

  def get_action_and_value(self, visual_obs, gamevars_obs, lstm_state, done, action=None):
    hidden, lstm_state = self.get_states(visual_obs, gamevars_obs, lstm_state, done)
    logits = self.actor(hidden)
    
    if self.multidiscrete_actions:
      split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
      multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
      if action is None:
        action = torch.stack([categorical.sample() for categorical in multi_categoricals])
      logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
      entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
      return action.T, logprob.sum(0), entropy.sum(0), self.critic(hidden), self._intrinsic_reward(visual_obs, gamevars_obs), lstm_state
    else:
      probs = Categorical(logits=logits)
      if action is None:
          action = probs.sample()
      return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), self._intrinsic_reward(visual_obs, gamevars_obs), lstm_state

  def _intrinsic_reward(self, visual_obs, gamevars_obs):
    pred_state = self.predictor_rnd_net(visual_obs, gamevars_obs)
    targ_state = self.target_rnd_net(visual_obs, gamevars_obs)
    return nn.functional.mse_loss(pred_state, targ_state.detach())
    

if __name__ == "__main__":
  args = parse_args()
  device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
  run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic

  # Setup the Gym environments for vizdoom
  envs = gym.vector.SyncVectorEnv(
    [make_env(args, args.seed + i, i, run_name) for i in range(args.num_envs)]
  )
  #envs = DoomNormalizeObservation(envs) # NOTE: This doesn't improve convergence and adds more overhead, not worth it
  
  # Create the run directory path for this training session
  run_dir = os.path.join("runs", args.gym_id, run_name)
  os.makedirs(run_dir, exist_ok=True)
  
  assert args.multidiscrete_actions and isinstance(envs.single_action_space, gym.spaces.MultiDiscrete) or \
    not args.multidiscrete_actions and isinstance(envs.single_action_space, gym.spaces.Discrete), \
    "Discrete/MultiDiscrete action space mismatch!"
    
  agent = Agent(args, envs).to(device)
  optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
  
  reward_i_normalizer = SimpleNormalizeReward(args.num_envs)
  
  # Storage setup: Keeps all the current horizon (args.num_steps) runs for each environment
  # at each update step. These values are then used for PPO after each environment has run for num_steps
  obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space[0].shape).to(device)
  gamevars = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space[1].shape).to(device)
  actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
  logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
  rewards_i = torch.zeros((args.num_steps, args.num_envs)).to(device) # intrinsic rewards
  rewards_e = torch.zeros((args.num_steps, args.num_envs)).to(device) # extrinsic rewards
  dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
  values_i_e = torch.zeros((args.num_steps, args.num_envs, 2)).to(device) # critic intrinsic, extrinsic values
  
  total_rewards_i = np.zeros((args.num_envs))      # Raw intrinsic rewards (for data tracking only)
  total_rewards_i_norm = np.zeros((args.num_envs)) # Normalized intrinsic rewards (for data tracking only)
  total_rewards_e_norm = np.zeros((args.num_envs)) # Normalized extrinsic rewards (for data tracking only)
  cum_scores = np.zeros(args.num_envs, dtype=np.float64) # Track cumulative scores for each environment (for data tracking only)

  global_step = 0
  init_step   = 0 
  num_updates = args.total_timesteps // args.batch_size
   
  # Load the checkpoint/model file if one was provided
  if args.model is not None and len(args.model) > 0:
    if os.path.exists(args.model):
      print(f"Model file '{args.model}' found, loading...")
      model_dict = torch.load(args.model)
      load_failed = False
      try:
        agent.load_state_dict(model_dict["agent"], strict=False)
      except RuntimeError as e:
        print("Could not load agent networks:")
        print(e)
        load_failed = True
      if not load_failed:
        optimizer.load_state_dict(model_dict["optim"])
        init_step = model_dict["timesteps"]
        global_step = init_step
        args.total_timesteps += global_step
        if "lr" in model_dict:
          args.learning_rate = model_dict["lr"]
        print("Model loaded!")
      else:
        print("Model loaded with failures.")
    else:
      print(f"Could not find/load model file '{args.model}'")

  if args.track:
    import wandb
    wandb.init(
      project=args.wandb_project_name,
      entity=args.wandb_entity,
      sync_tensorboard=True,
      config=vars(args),
      name=run_name,
      monitor_gym=True,
      save_code=True,
    )
    
  # Setup TensorBoard
  writer = SummaryWriter(run_dir)
  writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
  ) # Build a markdown table of all the hyperparameters / command line args

  next_obs_tuple = envs.reset()
  next_obs = torch.Tensor(next_obs_tuple[0]).to(device)
  next_gamevars = torch.Tensor(next_obs_tuple[1]).to(device)
  next_done = torch.zeros(args.num_envs).to(device)
  next_lstm_state = (
    torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
  )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
  
  
  # Exploration time!
  # For some fixed number of steps, we explore in each environment. This is needed
  # in order to pre-load the reward normalizer(s) so they are better suited to normalize
  # reward values when the time comes to start training
  print(f"Starting exploration for {args.num_explore_steps} steps...")
  with torch.no_grad():
    for _ in range(args.num_explore_steps):
      action, logprob, _, value_i_e, reward_i, next_lstm_state = agent.get_action_and_value(
        next_obs, next_gamevars, next_lstm_state, next_done
      )
      reward_i_normalizer.normalize(reward_i.cpu().numpy())
      next_obs_tuple, reward_e, done, info = envs.step(action.cpu().numpy())
      next_obs = torch.Tensor(next_obs_tuple[0]).to(device)
      next_gamevars = torch.Tensor(next_obs_tuple[1]).to(device)
      next_done = torch.Tensor(done).to(device)
  print("Finished exploration, starting training...")

  # Training time!
  start_time = time.time()
  lrnow = args.learning_rate
  for update in range(1, num_updates + 1):
      initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

      if args.anneal_lr:
        frac = 1.0 - 1.5*(update - 1.0) / (num_updates)
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

      for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        
        if (global_step // args.num_envs) % args.save_timesteps == 0:
          save_dict = {
            "timesteps": global_step,
            "agent": agent.state_dict(),
            "optim": optimizer.state_dict(),
            "lr": lrnow,
          }
          save_path = os.path.join(run_dir, f"doom_ppo_agent_{global_step}.chkpt")
          torch.save(save_dict, save_path)     
        
        obs[step] = next_obs
        gamevars[step] = next_gamevars
        dones[step] = next_done

        # Action logic
        with torch.no_grad():
          action, logprob, _, value_i_e, reward_i, next_lstm_state = agent.get_action_and_value(
            next_obs, next_gamevars, next_lstm_state, next_done
          )
          values_i_e[step] = value_i_e
          reward_i_np = reward_i.cpu().numpy()
          reward_i_np_norm = np.clip(reward_i_normalizer.normalize(reward_i_np) * args.reward_i_coeff, 0, 5)
          total_rewards_i += reward_i_np
          total_rewards_i_norm += reward_i_np_norm
          rewards_i[step] = torch.tensor(reward_i_np_norm).to(device)
        
        actions[step] = action
        logprobs[step] = logprob

        next_obs_tuple, reward_e, done, info = envs.step(action.cpu().numpy())
        
        total_rewards_e_norm += reward_e
        rewards_e[step] = torch.tensor(reward_e).to(device).view(-1)
        next_obs = torch.Tensor(next_obs_tuple[0]).to(device)
        next_gamevars = torch.Tensor(next_obs_tuple[1]).to(device)
        next_done = torch.Tensor(done).to(device)

        if global_step % 1000 == 0:
          writer.add_image("images/rgb_observation", next_obs[0,0:3], global_step)
          inc = 0
          if args.depth:
            writer.add_image("images/depth_observation", next_obs[0,3:4], global_step)
            inc += 1
          if args.labels:
            writer.add_image("images/labels_observation", next_obs[0,3+inc:4+inc], global_step)
            inc += 1
          if args.automap:
            writer.add_image("images/automap_observation", next_obs[0,3+inc:4+inc], global_step)
            inc += 1

        for env_idx, item in enumerate(info):
          if "episode" in item.keys():
            cum_scores[env_idx] += item["episode"]["r"]
            
            total_return = item["episode"]["r"] + total_rewards_i[env_idx]
            total_return_norm = total_rewards_i_norm[env_idx] + total_rewards_e_norm[env_idx]
            
            print(
              f"[{env_idx}] global_step={global_step}, episodic return: [" + 
              f"extrinsic: {item['episode']['r']:.2f}, intrinsic: {total_rewards_i[env_idx]:.2f}, total: {total_return:.2f}, " +
              f"norm extrinsic: {total_rewards_e_norm[env_idx]:.2f}, norm intrinsic: {total_rewards_i_norm[env_idx]:.2f}, norm total: {total_return_norm:.2f}]"
            )
            
            writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            writer.add_scalar("charts/cumulative_return", cum_scores[env_idx],  global_step)
            
            writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_return_intrinsic", total_rewards_i[env_idx], global_step)
            writer.add_scalar("charts/episodic_return_i+e", total_return, global_step)
            
            writer.add_scalar("charts/normalized/episodic_return_extrinsic", total_rewards_e_norm[env_idx], global_step)
            writer.add_scalar("charts/normalized/episodic_return_intrinsic", total_rewards_i_norm[env_idx], global_step)
            writer.add_scalar("charts/normalized/episodic_return", total_return_norm, global_step)
            
            total_rewards_i[env_idx] = 0
            total_rewards_i_norm[env_idx] = 0
            total_rewards_e_norm[env_idx] = 0
            
            break

      with torch.no_grad():
        # (Intrinsic, Extrinsic) value/reward
        advantages_i = torch.zeros_like(rewards_i).to(device)
        advantages_e = torch.zeros_like(rewards_e).to(device)
        lastgaelam_i = 0
        lastgaelam_e = 0
        values_i, values_e = torch.chunk(values_i_e, 2, dim=-1)
        for t in reversed(range(args.num_steps)):
          if t == args.num_steps - 1:
            next_value = agent.get_value(next_obs, next_gamevars, next_lstm_state, next_done)
            nextnonterminal = 1.0 - next_done
            next_value_i, next_value_e = torch.chunk(next_value, 2, dim=1)
            nextvalues_i = next_value_i.view(-1)
            nextvalues_e = next_value_e.view(-1)
          else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues_i = values_i[t + 1].view(-1)
            nextvalues_e = values_e[t + 1].view(-1)
          
          delta_i = rewards_i[t] + args.gamma_i * nextvalues_i - values_i[t].view(-1)
          advantages_i[t] = lastgaelam_i = delta_i + args.gamma_i * args.gae_lambda * lastgaelam_i # Don't let terminal states influence intrinsic rewards
          delta_e = rewards_e[t] + args.gamma_e * nextvalues_e * nextnonterminal - values_e[t].view(-1)
          advantages_e[t] = lastgaelam_e = delta_e + args.gamma_e * args.gae_lambda * nextnonterminal * lastgaelam_e
        
        returns_i = advantages_i + values_i.view(args.num_steps, -1)
        returns_e = advantages_e + values_e.view(args.num_steps, -1)

      # Flatten the batch
      b_obs = obs.reshape((-1,) + envs.single_observation_space[0].shape)
      b_gamevars = gamevars.reshape((-1,) + envs.single_observation_space[1].shape)
      b_logprobs = logprobs.reshape(-1)
      b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
      b_dones = dones.reshape(-1)
      
      b_returns_i    = returns_i.reshape(-1)
      b_returns_e    = returns_e.reshape(-1)
      b_values_i     = values_i.reshape(-1)
      b_values_e     = values_e.reshape(-1)
      b_advantages = (advantages_i + 2*advantages_e).reshape(-1)

      # Optimizing the policy and value network
      assert args.num_envs % args.num_minibatches == 0
      envsperbatch = args.num_envs // args.num_minibatches
      envinds = np.arange(args.num_envs)
      flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
      clipfracs = []
      for epoch in range(args.update_epochs):
          np.random.shuffle(envinds)
          for start in range(0, args.num_envs, envsperbatch):
              end = start + envsperbatch
              mbenvinds = envinds[start:end]
              mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

              _, newlogprob, entropy, newvalue_i_e, pt_loss, _ = agent.get_action_and_value(
                  b_obs[mb_inds], b_gamevars[mb_inds],
                  (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                  b_dones[mb_inds],
                  b_actions.long()[mb_inds].T if args.multidiscrete_actions else b_actions.long()[mb_inds],
              )
              logratio = newlogprob - b_logprobs[mb_inds]
              ratio = logratio.exp()

              with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

              mb_advantages = b_advantages[mb_inds]
              if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

              # Policy loss
              pg_loss1 = -mb_advantages * ratio
              pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
              pg_loss = torch.max(pg_loss1, pg_loss2).mean()

              # Value loss
              # NOTE: Keeping intrinsic and extrinsic value losses separated is crucial
              newvalue_i, newvalue_e = torch.chunk(newvalue_i_e, 2, dim=-1)
              newvalue_i = newvalue_i.reshape(-1)
              newvalue_e = newvalue_e.reshape(-1)
              
              v_loss_unclipped_i = (newvalue_i - b_returns_i[mb_inds]) ** 2
              v_clipped_i = b_values_i[mb_inds] + torch.clamp(newvalue_i - b_values_i[mb_inds], -args.clip_coef, args.clip_coef)
              v_loss_clipped_i = (v_clipped_i - b_returns_i[mb_inds]) ** 2
              v_loss_max_i = torch.max(v_loss_unclipped_i, v_loss_clipped_i)
              v_loss_i = 0.5 * v_loss_max_i.mean()
              
              v_loss_unclipped_e = (newvalue_e - b_returns_e[mb_inds]) ** 2
              v_clipped_e = b_values_e[mb_inds] + torch.clamp(newvalue_e - b_values_e[mb_inds], -args.clip_coef, args.clip_coef)
              v_loss_clipped_e = (v_clipped_e - b_returns_e[mb_inds]) ** 2
              v_loss_max_e = torch.max(v_loss_unclipped_e, v_loss_clipped_e)
              v_loss_e = 0.5 * v_loss_max_e.mean()
              v_loss = v_loss_i + v_loss_e
              
              entropy_loss = entropy.mean()
              pt_loss = pt_loss.mean()
              loss = pg_loss - entropy_loss * args.ent_coef + v_loss * args.vf_coef + pt_loss * args.pt_coef

              optimizer.zero_grad()
              loss.backward()
              nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
              optimizer.step()

          if args.target_kl is not None:
            if approx_kl > args.target_kl:
              break

      y_pred, y_true = (b_values_i+b_values_e).cpu().numpy(), (b_returns_i+b_returns_e).cpu().numpy()
      var_y = np.var(y_true)
      explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

      writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
      writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
      writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
      writer.add_scalar("losses/predictor_target_novelty_loss", pt_loss.item(), global_step)
      writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
      writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
      writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
      writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
      writer.add_scalar("losses/explained_variance", explained_var, global_step)
      
      steps_per_sec = int((global_step - init_step) / (time.time() - start_time))
      print(f"Steps per second: {steps_per_sec}")
      #writer.add_scalar("charts/Steps\/s", steps_per_sec, global_step)

  envs.close()
  writer.close()
  print("All training updates finished, exiting.")