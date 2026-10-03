import matplotlib.pyplot as plt
from RandomMatrix import *
N = 10
DysonMatrix = np.random.randn(N, N) * 0.01
x = WignersMatrix(N, 0.5)
simulation_time = 1000
step = 1/simulation_time
dyson_matrix_detector = []
for dt in range(simulation_time):
    DysonMatrix += np.array(x.sample().values) * np.sqrt(step)
    dyson_matrix_detector.append(DysonMatrix.copy())

eigenvalue_detector = []
for i in dyson_matrix_detector:
    eigenvalue_detector.append(np.linalg.eigvalsh(i))
    print(np.linalg.eigvalsh(i)[2])
for n in range(N):
    eigenvalue_records = [i[n] for i in eigenvalue_detector]
    plt.plot(np.array(range(0, simulation_time))/simulation_time, eigenvalue_records)
plt.title('eigenvalues-time')
plt.xlabel('time')
plt.ylabel('eigenvalues')
plt.show()
