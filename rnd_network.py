
import torch
import torch.nn as nn
from net_utils import layer_init_ortho

class RNDNetwork(nn.Module):
  def __init__(self, obs_shape, out_size, is_predictor) -> None:
    super(RNDNetwork, self).__init__()
    
    out_channel_list = [32, 64, 64]
    kernel_size_list = [ 8,  4,  3]
    stride_list      = [ 4,  2,  1]
    padding_list     = [ 0,  0,  0]

    self.out_size = out_size
    self.conv_net = nn.Sequential()
    _, curr_height, curr_width = obs_shape
    curr_channels = 3
    for out_channels, kernel_size, stride, padding in zip(out_channel_list, kernel_size_list, stride_list, padding_list):
      # NOTE: Square vs. Rectangular kernels appear to have no noticable effect
      # If anything, rectangular is worse. Use square kernels for simplicity.
      kernel_size_h = kernel_size
      kernel_size_w = kernel_size
      self.conv_net.append(layer_init_ortho(nn.Conv2d(curr_channels, out_channels, (kernel_size_h, kernel_size_w), stride)))
      self.conv_net.append(nn.LeakyReLU(0.2, inplace=True))
      
      curr_width  = int((curr_width-kernel_size_w + 2*padding) / stride + 1)
      curr_height = int((curr_height-kernel_size_h + 2*padding) / stride + 1)
      curr_channels = out_channels
        
    self.conv_net.append(nn.Flatten())
    conv_output_size = curr_width*curr_height*curr_channels
    
    # Concatenating game state variables into the inputs of each layer:
    # Orientation: 1 value in a [0,127] one-hot encoding
    # Position: 4 values each in a [0,255] one-hot encoding
    # Key Card pixels: 3x7x2 values in [0,1]
    gamevar_in_size = int(1*128 + 4*256 + 3*7*2)
    
    self.is_predictor = is_predictor
    if is_predictor:
      hidden_size = out_size#int(out_size * 1.5)
      self.fc_net1 = layer_init_ortho(nn.Linear(conv_output_size + gamevar_in_size, hidden_size))
      self.fc_net2 = layer_init_ortho(nn.Linear(hidden_size + gamevar_in_size, hidden_size))
      self.fc_net3 = layer_init_ortho(nn.Linear(hidden_size + gamevar_in_size, out_size))
    else:
      # Target network - this is just a random network that never gets trained
      self.fc_net = layer_init_ortho(nn.Linear(conv_output_size + gamevar_in_size, out_size))
      self.fc_net.eval()
      
  def forward(self, visual_obs, gamevars_obs):
    # Use one-hot encodings one each of the relevant game variables 
    var_tuples = torch.chunk(gamevars_obs, gamevars_obs.shape[-1], -1)
    one_hot_ori = nn.functional.one_hot(var_tuples[0].long(), num_classes=128).squeeze(1)
    one_hot_pos = torch.cat([nn.functional.one_hot(t.long(), num_classes=256).squeeze(1) for t in var_tuples[-4:]], -1)
    # The keycard pixel c, h, and w values are all flattened for concatenation
    keycard_pixels = visual_obs[:,0:3,53:60,60:62].flatten(1)
    
    def cat_gamevars(x):
      return torch.cat([x, one_hot_ori, one_hot_pos, keycard_pixels], -1)
      
    x = self.conv_net(visual_obs[:,0:3]) # Use rgb only for the visual observation
    if self.is_predictor:
      x = nn.functional.relu(self.fc_net1(cat_gamevars(x)), inplace=True)#nn.functional.leaky_relu(self.fc_net1(cat_gamevars(x)), 0.1, inplace=True)
      x = nn.functional.relu(self.fc_net2(cat_gamevars(x)), inplace=True)#nn.functional.leaky_relu(self.fc_net2(cat_gamevars(x)), 0.1, inplace=True)
      x = self.fc_net3(cat_gamevars(x))
    else:
      x = self.fc_net(cat_gamevars(x))
    
    return x
