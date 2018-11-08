import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from math import sqrt

import numpy as np

from layers.layer_norm import LayerNorm1d

### Code shamelessly stolen from jadore801120: github.com:/jadore801120/attention-is-all-you-need-pytorch
# Thanks man! :)

class Bottle(nn.Module):
  """ Perform the reshape routine before and after an operation"""

  def forward(self, x):
    if len(x.size()) <= 2:
      return super().forward(x)
    size = x.size()[:2]
    out = super().forward(x.view(size[0]*size[1], -1))
    return out.view(size[0], size[1], -1)

class BatchBottle(nn.Module):
  """ Perform the reshape routine before and after an operation """

  def forward(self, x):
    if len(x.size()) <= 2:
      return super().forward(x)
    size = x.size()[1:]
    out = super(BatchBottle, self).forward(x.view(-1, size[0]*size[1]))
    return out.view(-1, size[0], size[1])

class BottleSoftmax(Bottle, nn.Softmax):
  """ Perform the reshape routine before and after a softmax operation """
  pass

class BottleLayerNorm1d(BatchBottle, LayerNorm1d):
  """ Perform the reshape routine before and after a layer normalization """
  pass

class ScaledDotProductAttention(nn.Module):

  def __init__(self, d_h, drop_prob):
    super().__init__()
    self.drop_prob = drop_prob
    self.temper = sqrt(d_h)
    self.dropout = nn.Dropout(self.drop_prob)

  def forward(self, q, k, v, attn_mask = None):
    scores = q.bmm(k.transpose(1, 2)) / self.temper

    scores = self.dropout(scores)
    attn = F.softmax(scores, dim=2)
    output = attn.bmm(v)
    return output, attn

class MHA(nn.Module):

  def __init__(self, **kwargs):
    self.__dict__.update(**kwargs)
    super().__init__()
    self.build_module()

  def build_d_qk(self):
    return nn.Sequential(
      nn.Linear(self.d_model, self.d_k)
    )

  def build_d_v(self):
    return nn.Sequential(
      nn.Linear(self.d_model, self.d_v),
      nn.ReLU(),
    )

  def build_module(self):
    self.w_qs = self.build_d_qk()
    self.w_ks = self.build_d_qk()
    self.w_vs = self.build_d_v()

    self.attention = ScaledDotProductAttention(self.d_k // self.n_head, self.attn_drop_prob)

    self.layer_norm = LayerNorm1d(self.d_model)
    self.proj = nn.Linear(self.d_v, self.d_model)

  def forward(self, q, k, v, attn_mask = None):
    d_k, d_v = self.d_k, self.d_v
    n_head = self.n_head
    bs, n_tok, d_m = q.size()
    d_h = d_k // n_head

    residual = q
    h = self.layer_norm(q)


    qkv = [w_e(h).unsqueeze(1) for w_e in [self.w_qs, self.w_ks, self.w_vs]] #Bx1xTx(NH)
    qkv = [torch.cat(torch.chunk(e, n_head, dim=-1), 1) for e in qkv] #BxTxHxN
    qkv = [e.view(bs * n_head, n_tok, d_h) for e in qkv]

    h, attns = self.attention(*qkv)
    h = h.view(bs, -1, *h.size()[1:]).permute(0, 1, 2, 3).view(bs, n_tok, -1)
    outputs = self.proj(h)

    return outputs + residual, None






















































class MultiHeadAttention(nn.Module):

  def __init__(self, n_head, d_model, d_k, d_v, attn_drop_prob, drop_prob):
    super().__init__()
    self.n_head = n_head
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.attn_drop_prob = attn_drop_prob
    self.drop_prob = drop_prob

    self.w_qs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
    self.w_ks = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_k))
    self.w_vs = nn.Parameter(torch.FloatTensor(self.n_head, self.d_model, self.d_v))

    self.attention = ScaledDotProductAttention(
      d_model = self.d_k, drop_prob = self.attn_drop_prob
    )
    self.layer_norm = LayerNorm1d(self.d_model)
    self.proj = nn.Linear(self.n_head * self.d_v, self.d_model)

    self.dropout = nn.Dropout(self.drop_prob)

    self.reset_parameters()

  def reset_parameters(self):
    init.xavier_normal(self.w_qs)
    init.xavier_normal(self.w_ks)
    init.xavier_normal(self.w_vs)

  def forward(self, q, k, v, attn_mask = None):

    d_k, d_v = self.d_k, self.d_v
    n_head = self.n_head

    residual = q
    mb_size, len_q, d_q = q.size()
    mb_size, len_k, _ = k.size()
    mb_size, len_v, _ = v.size()

    #treat as a (n_head) size batch
    import pdb; pdb.set_trace()
    q_s = k_s = v_s = q.expand(n_head, -1, -1).view(n_head, -1, d_q).contiguous()
#    q_s, k_s, v_s = [e.repeat(n_head, 1, 1).view(n_head, -1, d_q) \
#                     for e in [q, k, v]]
    # treat the result as a (n_head * mb_size) size batch
    q_s, k_s, v_s = [e.bmm(w_e).view(-1, len_e, d_e) for (e, w_e, len_e, d_e) \
                     in [(q_s, self.w_qs, len_q, self.d_k), \
                         (k_s, self.w_ks, len_k, self.d_k), \
                         (v_s, self.w_vs, len_v, self.d_v)]]
    # perform attention, result size = (n_head * mb_size) * len_q * d_v
    outputs, attns = self.attention(
      q_s, k_s, v_s,
      None if attn_mask is None else attn_mask.repeat(n_head, 1, 1)
    )

    # back to original mb_size batch, result size = mb_size * len_q * (n_head * d_v)
    outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim = -1)

   # project back to residual size
    import pdb; pdb.set_trace()
    outputs = self.proj(outputs)
    outputs = self.dropout(outputs)

    return self.layer_norm(outputs + residual), attns

class PositionwiseFeedForward(nn.Module):

  def __init__(self, d_hid, d_inner_hid, drop_prob):
    super().__init__()
    self.d_hid = d_hid
    self.d_inner_hid = d_inner_hid
    self.drop_prob = drop_prob
    self.build_module()

  def build_module(self):
    self.stack = nn.Sequential(
      nn.Linear(self.d_hid, self.d_inner_hid),
      nn.ReLU(True),
      nn.Linear(self.d_inner_hid, self.d_hid),
      nn.Dropout(self.drop_prob)
    )
    self.layer_norm = LayerNorm1d(self.d_hid)

  def forward(self, x):
    h = self.stack(x)
    out = self.layer_norm(h + x)
    return out

class TransformerLayer(nn.Module):

  def __init__(
    self, n_head, d_model, d_k, d_v,
    mha_drop_prob, attn_drop_prob, d_ff, pos_ffn_drop_prob
  ):
    super().__init__()
    self.n_head = n_head
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.mha_drop_prob = mha_drop_prob
    self.attn_drop_prob = attn_drop_prob
    self.d_ff = d_ff
    self.pos_ffn_drop_prob = pos_ffn_drop_prob

    self.build_module()

  def build_module(self):
    self.slf_attn = MHA(
      n_head = self.n_head, d_model = self.d_model, d_k = self.d_k,
      d_v = self.d_v, drop_prob = self.mha_drop_prob,
      attn_drop_prob = self.attn_drop_prob
    )
#    self.pos_ffn = PositionwiseFeedForward(
#      d_hid = self.d_model, d_inner_hid = self.d_ff,
#      drop_prob = self.pos_ffn_drop_prob
#    )

  def forward(self, enc_input, slf_attn_mask = None):
    enc_output, enc_slf_attn = self.slf_attn(
      enc_input, enc_input, enc_input, attn_mask = slf_attn_mask
    )
#    enc_output = self.pos_ffn(enc_output)
    return enc_output, enc_slf_attn

class TransformerEncoder(nn.Module):

  def __init__(
    self, n_layers, n_head, d_model, d_k, d_v,
    mha_drop_prob, sdpa_drop_prob, d_ff, pos_ffn_drop_prob,
  ):
    super().__init__()
    self.n_layers = n_layers
    self.n_head = n_head
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.mha_drop_prob = mha_drop_prob
    self.sdpa_drop_prob = sdpa_drop_prob
    self.d_ff = d_ff
    self.pos_ffn_drop_prob = pos_ffn_drop_prob

    self.build_module()

  def build_module(self):
    self.c2t = Conv2Trans()
    self.t2c = Trans2Conv()
    self.layers = nn.ModuleList(
      [TransformerLayer(
        n_head = self.n_head,
        d_model = self.d_model,
        d_k = self.d_k, d_v = self.d_v,
        mha_drop_prob = self.mha_drop_prob,
        attn_drop_prob = self.sdpa_drop_prob,
        d_ff = self.d_ff,
        pos_ffn_drop_prob = self.pos_ffn_drop_prob
      ) for _ in range(self.n_layers)]
    )

  def forward(self, x, attn_mask = None, return_attns = False):
    out, attns = self.c2t(x), []
    for layer in self.layers:
      out, attn = layer(out, attn_mask)
      if return_attns:
        attns.append(attn)
    out = self.t2c(out)
    if return_attns:
      return out, attns
    else:
      return out

def position_encoding_init(n_position, d_pos_vec):
  """ initialize the sinusoid position encoding table """
  position_enc = np.array([
    [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)]
  )
  position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) #dim 2i
  position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) #dim 2i+1
  return torch.from_numpy(position_enc).type(torch.FloatTensor)

def img_pos_emb(x_size, y_size, emb_size):
  x_pos = position_encoding_init(x_size, emb_size // 2)
  y_pos = position_encoding_init(y_size, emb_size // 2)
  x_grid = x_pos.t().unsqueeze(2).expand(
    x_pos.size(1), x_pos.size(0), y_pos.size(0))
  y_grid = y_pos.t().unsqueeze(1).expand(
    y_pos.size(1), y_pos.size(0), x_pos.size(0))
  xy_grid = torch.cat([x_grid, y_grid], 0)
  return xy_grid

class Conv2Trans(nn.Module):

  def __init__(self):
    self.pos_emb = None
    super().__init__()

  def forward(self, x):
    if self.pos_emb is None:
      self.pos_emb = img_pos_emb(x.size(2), x.size(3), x.size(1)).cuda()
    with_pos = x + self.pos_emb
    flat = with_pos.view(x.size(0), x.size(1), -1).permute(0, 2, 1).contiguous()
    return flat

from math import sqrt

class Trans2Conv(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    perm = x.permute(0, 2, 1) # BxCx(HxW)
    h = w = int(sqrt(perm.size(2)))
    return perm.view(perm.size(0), perm.size(1), h, w).contiguous()
