import torch
import functools
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


from model import ScoreModel,autoencoder
from density import get_sample_batch_1d


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# For simplicity and as the example does it, we use the PDE dx = sigma^t dw, dw = Wiener

# Then, the marginal probability and diffusion coefficient are roughly
def marginal_prob_std(t, D):
  t = torch.tensor(t, device=device)
  return torch.sqrt(D * ( 1 - np.exp(-2*t)))

def diffusion_coeff(t, D):
  return torch.tensor(np.sqrt(2*D), device=device)


def loss_fn(model, x, marginal_prob_std, eps=1e-5,T=2):
  random_t = T*torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = (x*np.exp( - random_t)[:,None]) + (z * std[:, None])
  # perturbed_x = x + (z * std[:, None])
  score = model(perturbed_x, random_t)

  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))


  return loss


D = 1
marginal_prob_std_fn = functools.partial(marginal_prob_std, D=D)
diffusion_coeff_fn = functools.partial(diffusion_coeff, D=D)



# Define model
embed_dim = 128 # Dimension time is embedded in
dim = 1
model = ScoreModel(dim,embed_dim = embed_dim,marginal_prob_std=marginal_prob_std_fn)
# model = autoencoder(dim,embed_dim,marginal_prob_std_fn)

# Define optimizer
lr = 1e-4
momentum = 0.9
optimizer = optim.SGD(model.parameters(), lr=lr, momentum = momentum)


n_epochs =   5000
batch_size = 1024


losses = []
# torch.autograd.set_detect_anomaly(True)
for epoch in range(n_epochs):

    batch = get_sample_batch_1d(batch_size)
    batch.requires_grad = True
    loss = loss_fn(model,batch,marginal_prob_std_fn)


    if epoch % 100 == 0:
        print('{} out of {} epochs'.format(epoch,n_epochs))
        print('Loss: {}'.format(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())


torch.save(model.state_dict(),'model_weights.pth')

# plt.plot(range(n_epochs),losses)
# plt.show()
