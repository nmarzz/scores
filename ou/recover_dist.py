import torch
import functools
import numpy as np
from scipy import integrate
from model import ScoreModel,autoencoder
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                pdf_dim,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                device='cuda',
                z=None,
                eps=1e-3):

  t = torch.ones(batch_size, device=device)*2
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, pdf_dim, device=device) * marginal_prob_std(t)[:, None]
  else:
    init_x = z

  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
    with torch.no_grad():
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t, x):
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -x - 0.5 * (g**2)*score_eval_wrapper(x, time_steps)

  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (2, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device)

  return x


# Then, the marginal probability and diffusion coefficient are roughly
def marginal_prob_std(t, D):
  t = torch.tensor(t, device=device)
  return torch.sqrt(D * ( 1 - np.exp(-2*t)))

def diffusion_coeff(t, D):
  return torch.tensor(np.sqrt(2*D), device=device)


D = 2
marginal_prob_std_fn = functools.partial(marginal_prob_std, D=D)
diffusion_coeff_fn = functools.partial(diffusion_coeff, D=D)



weights = torch.load('model_weights.pth')
dim = 1
embed_dim = 128
model = ScoreModel(dim,embed_dim,marginal_prob_std_fn)
# model = autoencoder(dim,embed_dim,marginal_prob_std_fn)
model.load_state_dict(weights)


device = 'cpu'
error_tolerance = 1e-5
nsamples = 10000
results = ode_sampler(model,marginal_prob_std_fn,diffusion_coeff_fn,dim,atol=error_tolerance,rtol=error_tolerance,device=device,batch_size = nsamples)
results = results.numpy()


print(np.mean(results))
print(np.std(results))



plt.hist(results,density = True,bins = 50,stacked = False)
plt.title('Recovered density from learned score: T = 1e-3')
# mu = 0
# variance = 0.4643**2
# sigma = math.sqrt(variance)
# x = np.linspace(-4 + mu, 4 + mu, 100)
# plt.plot(x, stats.norm.pdf(x,0,sigma))
plt.xlabel('x')
plt.ylabel('density')
plt.show()






# with torch.no_grad():
#     device = 'cpu'
#     x = 0.1*torch.ones(10,1)
#     t = torch.tensor([0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889,
#             1.0000])
#     x = t.unsqueeze(1)
#     t =  2.3*torch.ones(10)
#
#     print(x)
#     print(t)
#     print(model(x,t))
#
#     plt.plot(x,model(x,t))
#     plt.show()


#
# with torch.no_grad():
#     x = torch.arange(-3, 3, 0.25)
#     t = torch.arange(0, 5, 0.25)
#     model(torch.tensor(2.).unsqueeze(0),torch.tensor(1.))
