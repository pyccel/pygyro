import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='Plot the l2 norm of phi as a function of time')
parser.add_argument('foldername', type=str, help='the name of the folder from which to load')

args = parser.parse_args()

data = np.loadtxt(os.path.join(args.foldername, 'phiDat.txt'))

t = data[:,0]
l2 = data[:,1]

plt.plot(t, 4e-5*np.exp(0.00354*t))
plt.semilogy(t,l2,'.')

plt.show()
