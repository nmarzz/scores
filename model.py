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

    def __init__(self,dimension,embed_dim,marginal_prob_std,sigma=1):
        super(ScoreModel,self).__init__()


        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                 nn.Linear(embed_dim, embed_dim))

        self.linear1 = nn.Linear(dimension,embed_dim)
        self.linear2 = nn.Linear(embed_dim,dimension)
        # self.linear2.weight = nn.Parameter(self.linear1.weight.transpose(0,1))

        self.dense1 = nn.Linear(embed_dim,embed_dim)
        self.dense2 = nn.Linear(embed_dim,embed_dim)

        self.sigma = sigma
        self.marginal_prob_std = marginal_prob_std

        self.swish = lambda x: x * torch.sigmoid(x)

    def forward(self,x,t):


        embed = self.swish(self.embed(t))

        h = self.linear1(x)
        h += self.dense1(embed)
        h = torch.sigmoid(h)
                
        h = self.linear2(h)

        h = (h-x) / self.marginal_prob_std(t)[:,None]

        return h



#
# class autoencoder(nn.Module):
#     def __init__(self,dim,embed_dim,marginal_prob_std):
#         super(autoencoder, self).__init__()
#         # Define time embedding
#         self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
#          nn.Linear(embed_dim, embed_dim))
#         self.marginal_prob_std = marginal_prob_std
#         self.swish = lambda x: x * torch.sigmoid(x)
#
#         self.linear1 = nn.Linear(dim,128)
#         self.dense1 = nn.Linear(embed_dim,128)
#         self.bnorm1d1 = nn.BatchNorm1d(128)
#         self.linear2 = nn.Linear(128,64)
#         self.dense2 = nn.Linear(embed_dim,64)
#         self.bnorm1d2 = nn.BatchNorm1d(64)
#         self.linear3 = nn.Linear(64,12)
#         self.dense3 = nn.Linear(embed_dim,12)
#         self.bnorm1d3 = nn.BatchNorm1d(12)
#         self.linear4 = nn.Linear(12,3)
#
#         self.tlinear4 = nn.Linear(3,12)
#         self.dense4 = nn.Linear(embed_dim,12)
#         self.bnorm1d4 = nn.BatchNorm1d(12)
#         self.tlinear3 = nn.Linear(12,64)
#         self.dense5 = nn.Linear(embed_dim,64)
#         self.bnorm1d5 = nn.BatchNorm1d(64)
#         self.tlinear2 = nn.Linear(64,128)
#         self.dense6 = nn.Linear(embed_dim,128)
#         self.bnorm1d6 = nn.BatchNorm1d(128)
#         self.tlinear1 = nn.Linear(128,dim)
#
#
#     def forward(self, x,t):
#         embed = self.swish(self.embed(t))
#
#         # Encode
#         h1 = self.linear1(x)
#         h1 += self.dense1(embed)
#         h1 = self.bnorm1d1(h1)
#         h1 = self.swish(h1)
#
#         h2 = self.linear2(h1)
#         h2 += self.dense2(embed)
#         h2 = self.bnorm1d2(h2)
#         h2 = self.swish(h2)
#
#         h3 = self.linear3(h2)
#         h3 += self.dense3(embed)
#         h3 = self.bnorm1d3(h3)
#         h3 = self.swish(h3)
#
#         h4 = self.linear4(h3)
#         h4 = self.swish(h4)
#
#         # Decode
#         h = self.tlinear4(h4)
#         h += self.dense4(embed)
#         h = self.bnorm1d4(h)
#         h = self.swish(h)
#
#         h = self.tlinear3(h)
#         h += self.dense5(embed)
#         h = self.bnorm1d5(h)
#         h = self.swish(h)
#
#         h = self.tlinear2(h)
#         h += self.dense6(embed)
#         h = self.bnorm1d6(h)
#         h = self.swish(h)
#
#         h = self.tlinear1(h)
#
#         h = h
#
#         return x


class autoencoder(nn.Module):
    def __init__(self,dim,embed_dim,marginal_prob_std):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, dim))

        self.marginal_prob_std = marginal_prob_std

    def forward(self, x,t):
        x = self.encoder(x)
        x = self.decoder(x)

        x = x / self.marginal_prob_std(t)[:,None]
        return x
