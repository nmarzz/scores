import torch
import numpy as np
from model import ScoreModel,autoencoder
import functools
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm


def marginal_prob_std(t, D):
  t = torch.tensor(t, device=device)
  return torch.sqrt(D * ( 1 - np.exp(-2*t)))
D = 1
marginal_prob_std_fn = functools.partial(marginal_prob_std, D=D)





weights = torch.load('model_weights.pth')
dim = 1
embed_dim = 128
score = ScoreModel(dim,embed_dim,marginal_prob_std_fn)
# model = autoencoder(dim,embed_dim,marginal_prob_std_fn)
score.load_state_dict(weights)
device = 'cpu'








fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
x = np.arange(-5, 5, 0.1)
t = np.arange(1e-3, 1, 0.1)
x, t = np.meshgrid(x, t)

r = np.sqrt(x**2 + t**2)

numrow = 0

with torch.no_grad():
    for i,xi in enumerate(x):
        if i ==1:
            break
        xi = torch.tensor(xi).float().unsqueeze(1)
        for j,tj in enumerate(t):
            tj = torch.tensor(tj).float()
            vals = score(xi,tj)
            vals = vals.numpy().transpose()

            if j == 0:
                score_vals = vals
            else:
                score_vals = np.vstack([score_vals, vals])
            numrow += 1



# print(x.shape)
# print(t.shape)
# print(score_vals.shape)
# print(score_vals)

# Plot the surface.
surf = ax.plot_surface(x, t, score_vals, cmap=cm.viridis)
# Add a color bar which maps values to colors.
plt.xlabel('x')
plt.ylabel('t')
plt.title('Learned Score p0_E')
plt.show()
