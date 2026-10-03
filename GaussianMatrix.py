import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_H(N=1000):
    gm = np.random.normal(size=(N, N))
    A = (gm + gm.T) / np.sqrt(2 * N)
    D = np.diag(np.sum(A, axis=1))
    L = D-A
    return L

def eigenvalues_spectrum(N=1000):
    H = generate_H(N)
    return np.linalg.eigvalsh(H)

def correlation_matrix(N=1000, samples=100):
    eiv = np.array([
        eigenvalues_spectrum(N)
        for _ in tqdm(range(samples))
    ])
    return np.corrcoef(eiv, rowvar=False)

def heatmap(N=1000, samples=400):
    cormatrix = correlation_matrix(N, samples)

    plt.imshow(
        cormatrix,
        cmap='viridis',
        aspect='auto',
        interpolation='nearest'
    )
    plt.colorbar(label='Correlation coefficient')
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue index')
    plt.title('Correlation Matrix heatmap')
    plt.show()

if __name__ == '__main__':
    heatmap()
