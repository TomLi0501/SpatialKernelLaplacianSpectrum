import numpy as np
import matplotlib.pyplot as plt
lam = []
n=32
for x in np.linspace(0, 2*np.pi,n):
    for y in np.linspace(0, 2*np.pi,n):
        lam.append(4 - 2*(np.cos(x) + np.cos(y)))
plt.vlines(lam,ymax=1, ymin=0)
plt.show()



