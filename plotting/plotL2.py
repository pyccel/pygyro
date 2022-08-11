import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('phiDat.txt')

t = data[:,0]
l2 = data[:,1]

plt.plot(t, 4e-5*np.exp(0.00354*t))
plt.semilogy(t,l2,'.')

plt.show()
