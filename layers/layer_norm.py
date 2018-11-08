import torch
from torch import nn
import torch.nn.functional as F

### Code shamelessly stolen from jadore801120: github.com:/jadore801120/attention-is-all-you-need-pytorch
# Thanks man! :)

class LayerNorm1d(nn.Module):
  """ Layer normalization module """

  def __init__(self, d_hid, eps=1e-3):
    super(LayerNorm1d, self).__init__()

    self.eps = eps
    self.scale = nn.Parameter(torch.ones(d_hid), requires_grad = True)
    self.bias = nn.Parameter(torch.zeros(d_hid), requires_grad = True)

  def forward(self, x):
    if x.size(1) == 1:
      return x

    mu = x.mean(keepdim=True, dim = -1)
    sigma = x.std(keepdim=True, dim = -1)
    ln_out = (x - mu.expand_as(x)) / (sigma.expand_as(x) + self.eps)
    ln_out = ln_out * self.scale.expand_as(ln_out) + self.bias.expand_as(ln_out)
    return ln_out
