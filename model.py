import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)







class ScoreModel(nn.Module):

    def __init__(self,dimension,embed_dim,sigma = 1):
        super(ScoreModel,self).__init__()

        self.linear1 = nn.Linear(dimension,embed_dim)
        self.linear2 = nn.Linear(embed_dim,dimension)


        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                 nn.Linear(embed_dim, embed_dim))

        self.dense1 = nn.Linear(embed_dim,embed_dim)
        self.dense2 = nn.Linear(embed_dim,embed_dim)


        self.sigma = sigma
        self.swish = lambda x: x * torch.sigmoid(x)

    def forward(self,x,t):
        # self.linear2.weight = nn.Parameter(self.linear1.weight.clone().transpose(0,1))

        embed = self.swish(self.embed(t))

        h = self.linear1(x)
        h += self.dense1(embed)

        # Maybe use batch norm?
        h = torch.sigmoid(h)
        # h += self.dense2(h)
        h = self.linear2(h)


        h = (h-x)/(self.sigma**2)

        return h
