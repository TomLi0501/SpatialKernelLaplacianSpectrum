import matplotlib.pyplot as plt
from RandomMatrix import *

def gaussian12():
    return np.random.normal(0, 1)

def gaussian3():
    return np.random.normal(0, np.sqrt(0.5))


def wigners(s):
    return s/2 * np.exp(-0.25 * s**2)

def semicircles(lam, sigma):
    return np.sqrt(4 * sigma**2 - lam**2) / (2 * np.pi * sigma**2)


if __name__ == '__main__':
    ''' Set up the experiment '''
    sample_size = 10000 # define number of matrix take
    shape = (2, 2) # shapes of the matrix
    m = RandomMatrix(shape) # now define the matrix generator
    m.set_variable('x1', gaussian12)
    m.set_variable('x2', gaussian12)
    m.set_variable('x3', gaussian3)
    m.set_matrix(np.array([['x1','x3'],['x3','x2']]))
    spacing_detector = []
    eigen_detector = []
    for _ in range(sample_size):
        spacing_detector = spacing_detector + m.sample().eigenvalue_spacing()
        eigen_detector = eigen_detector + list(m.sample().eigen().eigenvalues)
        # here both are normal list, so it will join together, not added.
    plt.hist(spacing_detector,bins=50, density=True) # density makes sure it is normalised
    xs = np.linspace(0, 7, 100)
    plt.plot(xs, wigners(xs), color='r')
    plt.title('Wigners Surmise')
    plt.show()

    plt.hist(eigen_detector,bins=50, density=True)
    xs = np.linspace(-7, 7, 100)
    sigma = np.sqrt(shape[0] * 0.5)
    plt.plot(xs, semicircles(xs, sigma), color='b')
    plt.title('Wigners Semicircles')
    plt.show()