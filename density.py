import torch # Use torch for ease of modelling in future
import numpy as np
import matplotlib.pyplot as plt
import sdeint
import sklearn.preprocessing

def base_density_1d(nsamples):
    ''' A function that returns n samples of some 'blackbox' base density '''
    dist1 = 10 + 1*torch.randn(nsamples//2)
    dist2 = -20 + 1*torch.randn(nsamples//2)

    samples = torch.cat((dist1,dist2))

    return samples


def get_sample_batch_1d(batch_size):
    samples = base_density_1d(batch_size)
    return samples.unsqueeze(1)



if __name__ == "__main__":
    # Sample from base density
    samples = base_density_1d(300).numpy()

    # Define an Ito SDE
    sigma = 100
    def f(x,t):
        return np.zeros(len(x))

    def g(x,t):
        return np.diag((sigma**t)*np.ones(len(x)))

    tspan = np.linspace(0.0, 1.0, 100)

    # Solve the SDE defined above
    result = sdeint.itoint(f,g,samples,tspan)

    plt.plot(tspan,result)
    plt.title('Particle paths')
    plt.show()

    plt.hist(samples,density = True)
    plt.title('Base density ')
    plt.show()

    plt.hist(result[-1],density = True)
    plt.title('Prior density ')
    plt.show()
