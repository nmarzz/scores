import torch
import functools
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


from model import ScoreModel
from density import get_sample_batch_1d


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# For simplicity and as the example does it, we use the PDE dx = sigma^t dw, dw = Wiener

# Then, the marginal probability and diffusion coefficient are roughly
def marginal_prob_std(t, sigma):
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  return torch.tensor(sigma**t, device=device)

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)

  perturbed_x = x + z * std[:, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))

  return loss


sigma =  25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


# Define model

embed_dim = 64 # Dimension time is embedded in
dim = 1
model = ScoreModel(dim,embed_dim = embed_dim)

# Define optimizer
lr = 1e-4
momentum = 0.9
optimizer = optim.SGD(model.parameters(), lr=lr, momentum = momentum)


n_epochs =   1500
batch_size = 256


losses = []
torch.autograd.set_detect_anomaly(True)
for epoch in range(n_epochs):
    print('{} out of {} epochs'.format(epoch,n_epochs))
    batch = get_sample_batch_1d(batch_size)
    loss = loss_fn(model,batch,marginal_prob_std_fn)
    print('Loss: {}'.format(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())


torch.save(model.state_dict(),'model_weights.pth')

plt.plot(range(n_epochs),losses)
plt.show()
