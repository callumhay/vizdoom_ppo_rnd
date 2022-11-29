import torch
import torch.nn as nn
import numpy as np

def layer_init_ortho(layer, std=np.sqrt(2), bias_const=0.0):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias_const)
  return layer

def fc_layer_init_rnd(layer, std=np.sqrt(2), bias_multiplier=0.01):
  nn.init.uniform_(layer.weight, -std, std)
  nn.init.uniform_(layer.bias,   -std*bias_multiplier, std*bias_multiplier)
  return layer

def fc_layer_init_norm(layer, std=np.sqrt(2)):
  nn.init.normal_(layer.weight, 0, std)
  nn.init.normal_(layer.bias, 0, std)
  with torch.no_grad():
    layer.bias *= 0.01
  return layer

def fc_layer_init_xavier(layer, nonlinearity, gain_arg=None):
  nn.init.xavier_normal_(layer.weight, nn.init.calculate_gain(nonlinearity, gain_arg))
  nn.init.normal_(layer.bias, 0, 1)
  with torch.no_grad(): layer.bias *= 0.01
  return layer

def conv_layer_init(layer, nonlinearity='linear', a=0, bias_const=True):
  nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity, a=a) # Change back to kaiming_normal?
  if bias_const:
    nn.init.constant_(layer.bias, 0)
  else:
    nn.init.uniform_(layer.bias, -0.01, 0.01)
  return layer

# Returns the rectangle (x,y,x+w,y+h) for finding the keycard pixels in the scaled down pixel/screenbuffer.
# NOTE: The returned rectangle is a inclusive min, exclusive max for use in ranges and array comprehension
# e.g., x:x+w / range(x,x+w) and y:y+h / range(y,y+h) should be used!
def keycard_rect(visual_obs_shape):
  start_h = int(53.0/60.0 * visual_obs_shape[-2])
  start_w = int(60.0/80.0 * visual_obs_shape[-1])
  end_w   = int(62.0/80.0 * visual_obs_shape[-1])
  return (start_w, start_h, end_w, visual_obs_shape[-2])

# This return a tensor of the cropped keycard pixels from the in-game HUD. This is super important
# information for getting the agent to properly explore a level that has key-locked doors.
# NOTE: If you change the screen size ratio for vizdoom, this may require modification - it relies
# on a ratio w:h of 4:3 (originally tested at RES_640X480).
def keycard_pixels_from_obs(visual_obs):
  start_w, start_h, end_w, end_h = keycard_rect(visual_obs.shape)
  return visual_obs[:, 0:3, start_h:end_h, start_w:end_w]