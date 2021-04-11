import torch # Use torch for ease of modelling in future
import numpy as np
import matplotlib.pyplot as plt
import sdeint
import sklearn.preprocessing

def base_density_1d(nsamples):
    ''' A function that returns n samples of some 'blackbox' base density '''
    dist1 =  5+ 1*torch.randn(nsamples//2)
    dist2 = -2 + 1*torch.randn(nsamples//2)

    samples = torch.cat((dist1,dist2))
    # samples = np.random.exponential(size = nsamples)
    # samples = torch.from_numpy(samples).float()

    # samples = torch.randn(nsamples)

    return samples


def get_sample_batch_1d(batch_size):
    samples = base_density_1d(batch_size)
    return samples.unsqueeze(1)



if __name__ == "__main__":
    # Sample from base density
    samples = base_density_1d(300).numpy()


    # Define an Ito SDE
    def f(x,t):
        return - x

    D = 1
    def g(x,t):
        return np.diag(np.sqrt(2*D)*np.ones(len(x)))
    T = 2
    tspan = np.linspace(0.0, T, 200)

    # Solve the SDE defined above
    result = sdeint.itoint(f,g,samples,tspan)

    plt.plot(tspan,result)
    plt.title('Particle paths')
    plt.show()

    plt.hist(samples,density = True)
    plt.title('True Bimodal Base Density')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.show()

    plt.hist(result[-1],density = True)
    plt.title('Prior density ')
    plt.show()


    print(np.mean(result[-1]))
    print(np.std(result[-1]))


    var = D * (1- np.exp(-2*T))
    print(np.sqrt(var))


    print(np.std(result[0]))
