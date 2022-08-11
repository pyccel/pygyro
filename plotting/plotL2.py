import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


def plot_file(foldername):
    dataset = np.atleast_2d(np.loadtxt(os.path.join(foldername, 'phiDat.txt')))

    shape = dataset.shape

    times = np.ndarray(shape[0])
    norm = np.ndarray(shape[0])

    times[:] = dataset[:, 0]
    norm[:] = dataset[:, 1]

    plt.semilogy(times, norm, '.', label=foldername)


parser = argparse.ArgumentParser(description='Process foldername')
parser.add_argument('-p', dest='pente', type=float,
                    default=3.83e-3,
                    help='The gradient of the expected linear regime (eg. for 1.12e-5*exp(3.83e-3*x) : 3.83e-3)')
parser.add_argument('-m', dest='mult', type=float,
                    default=1.12e-5,
                    help='The multiplication factor of the expected linear regime (eg. for 1.12e-5*exp(3.83e-3*x) : 1.12e-5)')
parser.add_argument('foldername', metavar='filename', nargs='*', type=str,
                    help='The folders whose results should be plotted')

args = parser.parse_args()
filename = os.path.join(args.foldername[0], 'phiDat.txt')

p = args.pente
m = args.mult

dataset = np.atleast_2d(np.loadtxt(filename))
sorted_times = np.sort(dataset[:, 0])

plt.figure()
plt.semilogy(sorted_times, m*np.exp(p*sorted_times),
             label=str(m)+'*exp('+str(p)+'*x)')
for f in args.foldername:
    plot_file(f)
plt.xlabel('time')
plt.ylabel('$\|\phi\|_2$')
plt.grid()
plt.legend()
plt.show()
