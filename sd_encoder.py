import torch
import torch.nn as nn
import numpy as np

from net_utils import layer_init_ortho, conv_layer_init

# Most of this code was originally taken and modified from the Stable Diffusion autoencoder:
# https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/autoencoder.py

def _normalize(in_channels, num_groups=16):
  return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def _nonlinearity(x):
  return x*torch.sigmoid(x)

class Upsample(nn.Module):
  def __init__(self, in_channels, with_conv):
    super().__init__()
    self.with_conv = with_conv
    if self.with_conv:
      self.conv = conv_layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))

  def forward(self, x):
    x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    if self.with_conv:
      x = self.conv(x)
    return x


class Downsample(nn.Module):
  def __init__(self, in_channels, with_conv):
    super().__init__()
    self.with_conv = with_conv
    if self.with_conv:
      self.conv = conv_layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0))

  def forward(self, x):
    if self.with_conv:
      # no asymmetric padding in torch conv, must do it ourselves
      pad = (0,1,0,1)
      x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
      x = self.conv(x)
    else:
      x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    return x

class ResnetBlock(nn.Module):
  def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout):
    super().__init__()
    self.in_channels = in_channels
    out_channels = in_channels if out_channels is None else out_channels
    self.out_channels = out_channels
    self.use_conv_shortcut = conv_shortcut

    self.norm1 = _normalize(in_channels)
    self.conv1 = conv_layer_init(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
    
    self.norm2 = _normalize(out_channels)
    self.dropout = torch.nn.Dropout(dropout)
    self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        self.conv_shortcut = conv_layer_init(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
      else:
        self.nin_shortcut = conv_layer_init(torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

  def forward(self, x):
    h = x
    h = self.norm1(h)
    h = _nonlinearity(h)
    h = self.conv1(h)

    h = self.norm2(h)
    h = _nonlinearity(h)
    h = self.dropout(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        x = self.conv_shortcut(x)
      else:
        x = self.nin_shortcut(x)

    return x+h

class AttnBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels

    self.norm = _normalize(in_channels)
    self.q = conv_layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))
    self.k = conv_layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))
    self.v = conv_layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))
    self.proj_out = conv_layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))


  def forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b,c,h,w = q.shape
    q = q.reshape(b,c,h*w)
    q = q.permute(0,2,1)   # b,hw,c
    k = k.reshape(b,c,h*w) # b,c,hw
    w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_ - w_.amax(keepdims=True), dim=2)

    # attend to values
    v = v.reshape(b,c,h*w)
    w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
    h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = h_.reshape(b,c,h,w)

    h_ = self.proj_out(h_)

    return x+h_
  
  
class SDEncoder(nn.Module):
  def __init__(self, args, envs):
    super().__init__()
    
    ch_mult = args.ch_mult
    dropout = 0.0 #args.dropout
    starting_channels = args.starting_channels
    z_channels = args.z_channels

    in_channels, res_h, res_w = envs.single_observation_space[0].shape
    self.num_resolutions = len(ch_mult)
    final_res = np.array([res_h, res_w]) // 2**(self.num_resolutions-1)
    final_fc_input_size = 2*z_channels*final_res[0]*final_res[1]
    net_output_size = args.net_output_size

    self.ch = starting_channels
    self.num_res_blocks = args.num_res_blocks
    self.res_block_ch_inds = set([1,2])#set(list(range(len(ch_mult)))) # Which levels the resnet(s) are present on
    self.in_channels = in_channels

    # downsampling
    self.conv_in = conv_layer_init(torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1))

    curr_res = np.array([res_h, res_w])
    in_ch_mult = (1,)+tuple(ch_mult)
    self.in_ch_mult = in_ch_mult
    self.down = nn.ModuleList()
    for i_level in range(self.num_resolutions):
      
      block_in = self.ch*in_ch_mult[i_level]
      block_out = self.ch*ch_mult[i_level]
      
      if i_level in self.res_block_ch_inds:
        block = nn.ModuleList()
        for _ in range(self.num_res_blocks):
          block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
          block_in = block_out
      else:
        block = None  

      down = nn.Module()
      down.block = block
      if i_level != self.num_resolutions-1:
        down.downsample = Downsample(block_in, True)
        curr_res = curr_res // 2
      self.down.append(down)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
    self.mid.attn_1  = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

    # end
    self.norm_out = _normalize(block_in)
    self.conv_out = conv_layer_init(torch.nn.Conv2d(block_in, 2*z_channels, kernel_size=3, stride=1, padding=1))
    self.final_layers = nn.Sequential(
      nn.ELU(inplace=True),
      nn.Flatten(),
      layer_init_ortho(nn.Linear(final_fc_input_size, net_output_size)),
      nn.LeakyReLU(0.25, inplace=True)
    )

  def forward(self, x):
    # downsampling
    hs = [self.conv_in(x)]
    for i_level in range(self.num_resolutions):
      curr_down = self.down[i_level]
      if curr_down.block is not None:
        for i_block in range(self.num_res_blocks):
          h = curr_down.block[i_block](hs[-1])
          hs.append(h)
      if i_level != self.num_resolutions-1:
        hs.append(curr_down.downsample(hs[-1]))

    # middle
    h = hs[-1]
    h = self.mid.block_1(h)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h)

    # end
    h = self.norm_out(h)
    h = _nonlinearity(h)
    h = self.conv_out(h)
    h = self.final_layers(h)
    return h
  